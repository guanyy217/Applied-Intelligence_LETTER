function [train_FC, test_FC] = LETTER_Update_FeatureCenters(Xp, k, train_data, test_data)

    num_train = size(train_data,1);
    num_test  = size(test_data,1);

    W = squareform(pdist(Xp'));

    Idx_p = SpectralClustering(W, k);
    FeatureIdx_p = cell(k,1);

    train_FC = zeros(num_train,k);
    test_FC  = zeros(num_test,k);

    for i = 1:k
        FeatureIdx_p{i,1} = find(Idx_p==i);
        mu = mean(Xp(:,FeatureIdx_p{i,1}));

        for n_train = 1:num_train
            train_FC(n_train,i) = dist(train_data(n_train,FeatureIdx_p{i,1}),mu);        
        end

        for n_test = 1:num_test
           test_FC(n_test,i)  = dist(test_data(n_test,FeatureIdx_p{i,1}),mu); 
        end  
    end
end

function D = dist(X,mu)
    a = X - mu;    
    b = a.^2;
    
    D = sqrt(sum(b,2))./size(X,2);        
end

function C = SpectralClustering(W,num_clusters)
    m = size(W, 1);
    
    sigma = abs(mean(mean(W)));
    S = exp(-(W.*W)./(2*sigma*sigma));
    
    %获得度矩阵D
    D = sparse(1:m, 1:m, sum(S));
    sqrt_pinv_D = sparse(1:m, 1:m, sqrt(1./sum(S)));
    
    % 获得拉普拉斯矩阵 Do laplacian, L = D^(-1/2) * (D-W) * D^(-1/2)
    L = full(sqrt_pinv_D * (D - S) * sqrt_pinv_D);
    
    % 求特征向量 V
    %  eigs 'SM';绝对值最小特征值
    [V, ~] = eigs(L, num_clusters, 'SM');
    
    if ~isreal(V)
        V = abs(V);
    end
    % 对特征向量求k-means
    C=kmeans(V,num_clusters);    
end

