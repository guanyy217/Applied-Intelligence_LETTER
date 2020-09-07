function [Outputs,Pre_Labels]=LETTER(train_data,train_target,test_data,test_target,ratio,svm,lambda1,lambda2,MaxIter)
    if(nargin<6)
        svm.type='Linear';
        svm.para=[];
    end
    
    if(nargin<5)
        ratio=0.1;
    end
    
    if(nargin<4)
        error('Not enough input parameters, please type "help LIFT" for more information');
    end
    
    [num_train,dim]=size(train_data);
    [num_class,num_test]=size(test_target);
    
    P_Centers=cell(num_class,1);
    N_Centers=cell(num_class,1);
    
%     P_ids=cell(num_class,1);
%     N_ids=cell(num_class,1);  

    Train_P_FC=cell(num_class,1);
    Train_N_FC=cell(num_class,1);
    
    Test_P_FC=cell(num_class,1);
    Test_N_FC=cell(num_class,1); 
 
    Idx_Feature = cell(num_class,1);
    
    %Find key instances of each label
    for i=1:num_class
        disp(['Performing clusteirng for the ',num2str(i),'/',num2str(num_class),'-th class']);
        
        p_idx=find(train_target(i,:)==1);
        n_idx=setdiff([1:num_train],p_idx);
        
        p_data=train_data(p_idx,:);
        n_data=train_data(n_idx,:); 
        
        
%         idx_p = find(any(p_data)~=0);
%         idx_n = find(any(n_data)~=0);        
%         
%         [Fp_entropy] = compute_entropy(p_data(:,idx_p)');
%         [Fn_entropy] = compute_entropy(n_data(:,idx_n)');
% 
%         [~,I_Fp_entropy] = sort(Fp_entropy,'descend'); 
%         [~,I_Fn_entropy] = sort(Fn_entropy,'descend'); 
% 
%         num_select_feature = min(round(max(length(idx_p),length(idx_n))*0.5),500);
% 
%         if length(idx_p) < num_select_feature
%             idx_Fp = idx_p';
%         else
%             idx_Fp = I_Fp_entropy(1:num_select_feature);
%         end
%         idx_Fn = I_Fn_entropy(1:num_select_feature);
%         Idx_Feature{i,1} = [idx_Fp;idx_Fn]; 
%         Idx_Feature{i,1} = unique(Idx_Feature{i,1});
% 
%         p_data = p_data(:,Idx_Feature{i,1});
%         n_data = n_data(:,Idx_Feature{i,1});
           
        Idx_Feature{i,1} = 1:dim;
        
        
        k_i=min(ceil(length(p_idx)*ratio),ceil(length(n_idx)*ratio));     
        
        if length(p_idx)~=1
            pf_idx = find(any(p_data)~=0);
        else
            pf_idx = find(p_data~=0);
        end
        nf_idx = find(any(n_data)~=0);
        k_f=min(ceil(length(pf_idx)*ratio),ceil(length(nf_idx)*ratio));
        k_f=min(50,k_f);
        
        pids = [];
        nids = [];
        
        if(k_i==0)
            POS_C=[];
%             NEG_C = train_data(N_ids{i,1},:);
            [~,NEG_C]=kmeans(train_data,min(50,num_train),'EmptyAction','singleton','OnlinePhase','off','Display','off');
        else
            if(size(p_data,1)==1)
                POS_C=p_data;
            else
                [~,POS_C]=kmeans(p_data,k_i,'EmptyAction','singleton','OnlinePhase','off','Display','off');
            end
            
            if(size(n_data,1)==1)
                NEG_C=n_data;
            else
%                 NEG_C = train_data(N_ids{i,1},:);
                [~,NEG_C]=kmeans(n_data,k_i,'EmptyAction','singleton','OnlinePhase','off','Display','off');
            end
        end  
        
        
%         if(k_i==0)
%             [POS_C,NEG_C] = LETTER_Update_InstanceCenters([], 0, pids, train_data, min(50,num_train), nids,lambda1,lambda2,MaxIter,'EmptyAction','singleton','OnlinePhase','off','Display','off');                
%         else 
%             [POS_C,NEG_C] = LETTER_Update_InstanceCenters(p_data, k_i, pids, n_data, k_i, nids,lambda1,lambda2,MaxIter,'EmptyAction','singleton','OnlinePhase','off','Display','off');           
%             
%             if(size(p_data,1)==1)
%                 POS_C=p_data;
%             end
%             
%             if(size(n_data,1)==1)
%                 NEG_C=n_data;
%             end                       
%         end
           
        if(k_f==0) | length(p_idx)==1
            [train_N_FC, test_N_FC] = LETTER_Update_FeatureCenters(n_data(:,nf_idx), min(50,ceil(length(nf_idx)*ratio)), train_data, test_data);     
        else 
            [train_P_FC, test_P_FC] = LETTER_Update_FeatureCenters(p_data(:,pf_idx), k_f, train_data, test_data);
            [train_N_FC, test_N_FC] = LETTER_Update_FeatureCenters(n_data(:,nf_idx), k_f, train_data, test_data);                      
        end
        
        P_Centers{i,1}=POS_C;
        N_Centers{i,1}=NEG_C; 
        
        Train_P_FC{i,1}=train_P_FC;
        Train_N_FC{i,1}=train_N_FC;     
        
        Test_P_FC{i,1}=test_P_FC;
        Test_N_FC{i,1}=test_N_FC;            
    end
    
    switch svm.type
        case 'RBF'
            gamma=num2str(svm.para);
            str=['-t 2 -g ',gamma,' -b 1'];
        case 'Poly'
            gamma=num2str(svm.para(1));
            coef=num2str(svm.para(2));
            degree=num2str(svm.para(3));
            str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
        case 'Linear'
            str='-t 0 -b 1';
        otherwise
            error('SVM types not supported, please type "help LIFT" for more information');
    end
    
    Models=cell(num_class,1);
    
    %Perform representation transformation and training
    for i=1:num_class        
        disp(['Building classifiers: ',num2str(i),'/',num2str(num_class)]);
        
        centers=[P_Centers{i,1};N_Centers{i,1}];
        num_center=size(centers,1);
        
        data=[];
        
        if(num_center>=5000)
            error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
        else
            blocksize=5000-num_center;
            num_block=ceil(num_train/blocksize);
            for j=1:num_block-1
                low=(j-1)*blocksize+1;
                high=j*blocksize;
                
                tmp_mat=[centers;train_data(low:high,Idx_Feature{i,1})];
                Y=pdist(tmp_mat);
                Z=squareform(Y);
                data=[data;Z((num_center+1):(num_center+blocksize),1:num_center)];                
            end
            
            low=(num_block-1)*blocksize+1;
            high=num_train;
            
            tmp_mat=[centers;train_data(low:high,Idx_Feature{i,1})];
            Y=pdist(tmp_mat);
            Z=squareform(Y);

            data=[data;Z((num_center+1):(num_center+high-low+1),1:num_center)];

        end
        
        training_instance_matrix=[data,Train_P_FC{i,1},Train_N_FC{i,1}];
        training_label_vector=train_target(i,:)';

        Models{i,1}=svmtrain(training_label_vector,training_instance_matrix,str);      
    end   

    %Perform representation transformation and testing
    Pre_Labels=[];
    Outputs=[];
    for i=1:num_class        
        centers=[P_Centers{i,1};N_Centers{i,1}];
        num_center=size(centers,1);
        
        data=[];
        
        if(num_center>=5000)
            error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
        else
            blocksize=5000-num_center;
            num_block=ceil(num_test/blocksize);
            for j=1:num_block-1
                low=(j-1)*blocksize+1;
                high=j*blocksize;
                
                tmp_mat=[centers;test_data(low:high,Idx_Feature{i,1})];
                Y=pdist(tmp_mat);
                Z=squareform(Y);
                data=[data;Z((num_center+1):(num_center+blocksize),1:num_center)];                
            end
            
            low=(num_block-1)*blocksize+1;
            high=num_test;
            
            tmp_mat=[centers;test_data(low:high,Idx_Feature{i,1})];
            Y=pdist(tmp_mat);
            Z=squareform(Y);
            data=[data;Z((num_center+1):(num_center+high-low+1),1:num_center)];

        end
        
        testing_instance_matrix=[data,Test_P_FC{i,1},Test_N_FC{i,1}];
        testing_label_vector=test_target(i,:)';    
        
        [predicted_label,accuracy,prob_estimates]=svmpredict(testing_label_vector,testing_instance_matrix,Models{i,1},'-b 1');
        if(isempty(predicted_label))
            predicted_label=train_target(i,1)*ones(num_test,1);
            if(train_target(i,1)==1)
                Prob_pos=ones(num_test,1);
            else
                Prob_pos=zeros(num_test,1);
            end
            Outputs=[Outputs;Prob_pos'];
%             Pre_Labels=[Pre_Labels;predicted_label'];
        else
            pos_index=find(Models{i,1}.Label==1);
            Prob_pos=prob_estimates(:,pos_index);
            Outputs=[Outputs;Prob_pos'];
%             Pre_Labels=[Pre_Labels;predicted_label'];
        end
    end
    
    [O,I] = sort(Outputs,'descend');
    for i=1:num_test
        l = sum(test_target(:,i)==1);

        Pre_Labels(I(1:l,i),i) = 1;
        Pre_Labels(I(l+1:num_class,i),i) = -1;
    end

    