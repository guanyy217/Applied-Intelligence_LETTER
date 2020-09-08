function varargout = LETTER_Update_InstanceCenters(Xp, Xn, k, lambda1, lambda2, MaxIter, varargin)
if nargin < 2
    error(message('stats:kmeans:TooFewInputs'));
end

Xp = norm(Xp);
Xn = norm(Xn);

% n points in p dimensional space
[np] = size(Xp,1);
[nn] = size(Xn,1);

pnames = {   'distance'  'start' 'replicates' 'emptyaction' 'onlinephase' 'options' 'maxiter' 'display'};
dflts =  {'sqeuclidean' 'plus'          []  'singleton'         'off'        []        []        []};
[distance,start,reps,emptyact,online,options,maxit,display] ...
    = internal.stats.parseArgs(pnames, dflts, varargin{:});

distNames = {'sqeuclidean','cityblock','cosine','correlation','hamming'};
distance = internal.stats.getParamVal(distance,distNames,'''Distance''');

emptyactNames = {'error','drop','singleton'};
emptyact = internal.stats.getParamVal(emptyact,emptyactNames,'''EmptyAction''');

[~,online] = internal.stats.getParamVal(online,{'on','off'},'''OnlinePhase''');
online = (online==1);

% 'maxiter' and 'display' are grandfathered as separate param name/value pairs
if ~isempty(display)
    options = statset(options,'Display',display);
end
if ~isempty(maxit)
    options = statset(options,'MaxIter',maxit);
end

options = statset(statset('kmeans'), options);
display = find(strncmpi(options.Display, {'off','notify','final','iter'},...
    length(options.Display))) - 1;
maxit = options.MaxIter;

if ~(isscalar(k) && isnumeric(k) && isreal(k) && k > 0 && (round(k)==k))
    error(message('stats:kmeans:InvalidK'));
    % elseif k == 1
    % this special case works automatically
elseif np < k
    error(message('stats:kmeans:TooManyClusters'));
end

% Assume one replicate
if isempty(reps)
    reps = 1;
elseif ~internal.stats.isScalarInt(reps,1)
    error(message('stats:kmeans:BadReps'));
end

[useParallel, RNGscheme, poolsz] = ...
    internal.stats.parallel.processParallelAndStreamOptions(options,true);

usePool = useParallel && poolsz>0;

emptyErrCnt = 0;

% Define the function that will perform one iteration of the
% loop inside smartFor
loopbody = @loopBody;

% Initialize nested variables so they will not appear to be functions here
p_totsumD = 0;
n_totsumD = 0;
iter = 0;

Xp = Xp'; %Transpose data into column orientation
Xn = Xn'; %Transpose data into column orientation


% Perform KMEANS replicates on separate workers.
ClusterBest = internal.stats.parallel.smartForReduce(...
    reps, loopbody, useParallel, RNGscheme, 'argmin');

% Extract the best solution
varargout{1} = ClusterBest{1}';
varargout{2} = ClusterBest{2}';

if display > 1 % 'final' or 'iter'
    fprintf('%s\n',getString(message('stats:kmeans:FinalSumOfDistances',sprintf('%g',totsumDbest))));
end
 
% if hadNaNs
%     varargout{1} = statinsertnan(wasnan, varargout{1});% idxbest 
%     if nargout > 3
%         varargout{4} = statinsertnan(wasnan, varargout{4}); %Dbest
%     end
% end

    function cellout = loopBody(rep,S)
        
        if isempty(S)
            S = RandStream.getGlobalStream;
        end
        
        cellout = cell(4,1);        
        
        switch start
            case {'plus','kmeans++'}
                % Select the first seed by sampling uniformly at random
                index = zeros(1,k);
                [Cp(:,1), index(1)] = datasample(S,Xp,1,2);
                [Cn(:,1), index(1)] = datasample(S,Xn,1,2);
                p_minDist = inf(np,1);
                n_minDist = inf(nn,1);
           
                % Select the rest of the seeds by a probabilistic model
               for ii = 2:k                    
                    P_minDist = min(p_minDist,distfun(Xp,Cp(:,ii-1),distance));
                    N_minDist = min(n_minDist,distfun(Xn,Cn(:,ii-1),distance));
                    P_denominator = sum(P_minDist);
                    if P_denominator==0 || isinf(P_denominator) || isnan(P_denominator)
                        Cp(:,ii:k) = datasample(S,Xp,k-ii+1,2,'Replace',false);
                        break;
                    end
                    N_denominator = sum(N_minDist);
                    if N_denominator==0 || isinf(N_denominator) || isnan(N_denominator)
                        Cn(:,ii:k) = datasample(S,Xn,k-ii+1,2,'Replace',false);
                        break;
                    end
                    P_sampleProbability = P_minDist/P_denominator;
                    [Cp(:,ii), index(ii)] = datasample(S,Xp,1,2,'Replace',false,...
                        'Weights',P_sampleProbability);        
                    N_sampleProbability = N_minDist/N_denominator;
                    [Cn(:,ii), index(ii)] = datasample(S,Xn,1,2,'Replace',false,...
                        'Weights',N_sampleProbability);     
                end
        end
        if ~isfloat(Cp)      % X may be logical
            Cp = double(Cp);
        end
        if ~isfloat(Cn)      % X may be logical
            Cn = double(Cn);
        end
        
        [p_d, p_idx, p_m] = init_assignment(Xp,Cp,k,distance);
        [n_d, n_idx, n_m] = init_assignment(Xn,Cn,k,distance);
        
        try % catch empty cluster errors and move on to next rep            
            % Begin phase one:  batch reassignments
            converged = batchUpdate();            
            
            if ~converged
                if reps==1
                    warning(message('stats:kmeans:FailedToConverge', maxit));
                else
                    warning(message('stats:kmeans:FailedToConvergeRep', maxit, rep));
                end
            end            
          
            % Save the best solution so far
            cellout = {Cp,Cn}';
           
            % If an empty cluster error occurred in one of multiple replicates, catch
            % it, warn, and move on to next replicate.  Error only when all replicates
            % fail.  Rethrow an other kind of error.
        catch ME
            if reps == 1 || (~isequal(ME.identifier,'stats:kmeans:EmptyCluster')  && ...
                         ~isequal(ME.identifier,'stats:kmeans:EmptyClusterRep'))
                rethrow(ME);
            else
                emptyErrCnt = emptyErrCnt + 1;
                warning(message('stats:kmeans:EmptyClusterInBatchUpdate', rep, iter));
                if emptyErrCnt == reps
                    error(message('stats:kmeans:EmptyClusterAllReps'));
                end
            end
        end % catch
        
        %------------------------------------------------------------------
        
        function converged = batchUpdate()
            
            % Every point moved, every cluster will need an update
            p_moved = 1:np;
            p_changed = 1:k;
            p_previdx = zeros(np,1);
            p_prevtotsumD = Inf;
            
            n_moved = 1:nn;
            n_changed = 1:k;
            n_previdx = zeros(nn,1);
            n_prevtotsumD = Inf;
            
            %
            % Begin phase one:  batch reassignments
            %
            
            iter = 0;
            converged = false;
            while true
                iter = iter + 1;
                
                % Calculate the new cluster centroids and counts, and update the
                % distance from every point to those new cluster centroids
                [Cp(:,p_changed), p_m(p_changed)] = gcentroids(Xp, p_idx, p_changed, distance);
                Cp = Softthres(Cp, lambda1/2);
                Dp(:,p_changed) = distfun(Xp, Cp(:,p_changed), distance);
                
                [Cn(:,n_changed), n_m(n_changed)] = gcentroids(Xn, n_idx, n_changed, distance);
                Cn = UpdateCn(Cn,Cp,lambda2);
                Dn(:,n_changed) = distfun(Xn, Cn(:,n_changed), distance);
                
                % Deal with clusters that have just lost all their members
                p_empties = p_changed(p_m(p_changed) == 0);
                n_empties = n_changed(n_m(n_changed) == 0);
                if ~isempty(p_empties)
                    if strcmp(emptyact,'error')
                        if reps==1
                            error(message('stats:kmeans:EmptyCluster', iter));
                        else
                            error(message('stats:kmeans:EmptyClusterRep', iter, rep));
                        end
                    end
                    switch emptyact
                        case 'drop'
                            if reps==1
                                warning(message('stats:kmeans:EmptyCluster', iter));
                            else
                                warning(message('stats:kmeans:EmptyClusterRep', iter, rep));
                            end
                            % Remove the empty cluster from any further processing
                            Dp(:,p_empties) = NaN;
                            Dn(:,n_empties) = NaN;
                            p_changed = p_changed(p_m(p_changed) > 0);
                            n_changed = n_changed(n_m(n_changed) > 0);
                        case 'singleton'
                            [Cp, Dp, p_m, p_changed] = Temp(Xp, Cp, p_empties, Dp, p_idx, p_m, np, distance, p_changed);
                            [Cn, Dn, n_m, n_changed] = Temp(Xn, Cn, n_empties, Dn, n_idx, n_m, nn, distance, n_changed);
                    end
                end
                
                % Compute the total sum of distances for the current configuration.
                p_totsumD = sum(Dp((p_idx-1)*np + (1:np)'));
                n_totsumD = sum(Dn((n_idx-1)*nn + (1:nn)'));
                % Test for a cycle: if objective is not decreased, back out
                % the last step and move on to the single update phase
                if p_prevtotsumD <= p_totsumD
                    p_idx = p_previdx;
                    [Cp(:,p_changed), p_m(p_changed)] = gcentroids(Xp, p_idx, p_changed, distance);
                    iter = iter - 1;
                    break;
                end
                
                if n_prevtotsumD <= n_totsumD
                    n_idx = n_previdx;
                    [Cn(:,n_changed), n_m(n_changed)] = gcentroids(Xn, n_idx, n_changed, distance);
                    iter = iter - 1;
                    break;
                end
                
                if iter >= maxit
                    break;
                end
                
                % Determine closest cluster for each point and reassign points to clusters
                p_previdx = p_idx;
                p_prevtotsumD = p_totsumD;
                [p_d, p_nidx] = min(Dp, [], 2);
                
                n_previdx = n_idx;
                n_prevtotsumD = n_totsumD;
                [n_d, n_nidx] = min(Dn, [], 2);
                
                % Determine which points moved
                p_moved = find(p_nidx ~= p_previdx);
                if ~isempty(p_moved)
                    % Resolve ties in favor of not moving
                    p_moved = p_moved(Dp((p_previdx(p_moved)-1)*np + p_moved) > p_d(p_moved));
                end
                n_moved = find(n_nidx ~= n_previdx);
                if ~isempty(n_moved)
                    % Resolve ties in favor of not moving
                    n_moved = n_moved(Dn((n_previdx(n_moved)-1)*nn + n_moved) > n_d(n_moved));
                end
                
                if isempty(p_moved) || isempty(n_moved)
                    converged = true;
                    break;
                end
                
                p_idx(p_moved) = p_nidx(p_moved);
                n_idx(n_moved) = n_nidx(n_moved);
                
                % Find clusters that gained or lost members
                p_changed = unique([p_idx(p_moved); p_previdx(p_moved)])';
                n_changed = unique([n_idx(n_moved); n_previdx(n_moved)])';
                
            end % phase one            
        end % nested function       
    end

end % main function

%------------------------------------------------------------------

function X = norm(X)
    if ~isreal(X)
        error(message('stats:kmeans:ComplexData'));
    end
    wasnan = any(isnan(X),2);
    hadNaNs = any(wasnan);
    if hadNaNs
        warning(message('stats:kmeans:MissingDataRemoved'));
        X = X(~wasnan,:);
    end
end

function [Xmaxs, Xmins] = max_min(X,k,start,reps)
    Xmins = [];
    Xmaxs = [];
    CC = [];
    if ischar(start)
        startNames = {'uniform','sample','cluster','plus','kmeans++'};
        j = find(strncmpi(start,startNames,length(start)));
        if length(j) > 1
            error(message('stats:kmeans:AmbiguousStart', start));
        elseif isempty(j)
            error(message('stats:kmeans:UnknownStart', start));
        elseif isempty(k)
            error(message('stats:kmeans:MissingK'));
        end
        start = startNames{j};
        if strcmp(start, 'uniform')
            if strcmp(distance, 'hamming')
                error(message('stats:kmeans:UniformStartForHamm'));
            end
            Xmins = min(X,[],1);
            Xmaxs = max(X,[],1);
        end
    elseif isnumeric(start)
        CC = start;
        start = 'numeric';
        if isempty(k)
            k = size(CC,1);
        elseif k ~= size(CC,1)
            error(message('stats:kmeans:StartBadRowSize'));
        elseif size(CC,2) ~= p
            error(message('stats:kmeans:StartBadColumnSize'));
        end
        if isempty(reps)
            reps = size(CC,3);
        elseif reps ~= size(CC,3)
            error(message('stats:kmeans:StartBadThirdDimSize'));
        end
    else
        error(message('stats:kmeans:InvalidStart'));
    end
end

function [d,idx,m] = init_assignment(X,C,k,distance)
	% Compute the distance from every point to each cluster centroid and the
	% initial assignment of points to clusters
	D = distfun(X, C, distance);
	[d, idx] = min(D, [], 2);
	m = accumarray(idx,1,[k,1])';
end

function Cn = UpdateCn(Cn,Cp,lambda2)
    if ~isempty(Cp)
        Cn = Cn - lambda2.*mean(Cp,2);
    end    
    Cn = Softthres(Cn,0);
end

function [C, D, m, changed] = Temp(X, C, empties, D, idx, m, n, distance, changed)
    for i = empties
        d = D((idx-1)*n + (1:n)'); % use newly updated distances

        % Find the point furthest away from its current cluster.
        % Take that point out of its cluster and use it to create
        % a new singleton cluster to replace the empty one.
        [~, lonely] = max(d);
        from = idx(lonely); % taking from this cluster
        if m(from) < 2
            % In the very unusual event that the cluster had only
            % one member, pick any other non-singleton point.
            from = find(m>1,1,'first');
            lonely = find(idx==from,1,'first');
        end
        C(:,i) = X(:,lonely);
        m(i) = 1;
        idx(lonely) = i;
        D(:,i) = distfun(X, C(:,i), distance);

        % Update clusters from which points are taken
        [C(:,from), m(from)] = gcentroids(X, idx, from, distance);
        D(:,from) = distfun(X, C(:,from), distance);
        changed = unique([changed from]);
    end
end

function D = distfun(X, C, dist)
%DISTFUN Calculate point to cluster centroid distances.
n_X = size(X,2);
n_C = size(C,2);
switch dist
    case 'sqeuclidean'
        tmp_mat=[X,C]';
        Y=pdist(tmp_mat);
        Z=squareform(Y);
        D = Z(1:n_X,(n_X+1):(n_X+n_C));
%         D = pdist2mex(X,C,'sqe',[],[],[]);
    end
end % function

%------------------------------------------------------------------
function [centroids, counts] = gcentroids(X, index, clusts, dist)
%GCENTROIDS Centroids and counts stratified by group.
p = size(X,1);
num = length(clusts);

centroids = NaN(p,num,'like',X);
counts = zeros(1,num,'like',X);

for i = 1:num
    members = (index == clusts(i));
    if any(members)
       counts(i) = sum(members);
       switch dist
           case {'sqeuclidean','cosine','correlation'}
              centroids(:,i) = sum(X(:,members),2) / counts(i);
      end
    end
end
end % function
