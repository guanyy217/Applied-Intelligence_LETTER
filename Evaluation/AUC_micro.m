function AUCmicro=AUC_micro(Outputs,test_target)
%Computing the hamming loss
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance]=size(Outputs);
    temp_Outputs=[];
    temp_test_target=[];
    for i=1:num_instance
        temp=test_target(:,i);
        if((sum(temp)~=num_class)&(sum(temp)~=-num_class))
            temp_Outputs=[temp_Outputs,Outputs(:,i)];
            temp_test_target=[temp_test_target,temp];
        end
    end
    Outputs=temp_Outputs;
    test_target=temp_test_target;     
    [num_class,num_instance]=size(Outputs);
    
%     Relevant=cell(num_class*num_instance,1);
%     not_Relevant=cell(num_class*num_instance,1);
    Outputs = Outputs(:);
    test_target = test_target(:);
	
    
	
    Relevant = find(test_target==1);
    not_Relevant = find(test_target~=1);
       
    Relevant_pair_size = length(Relevant);
    not_Relevant_pair_size = num_class*num_instance - Relevant_pair_size;
        
    temp = 0;
    for m = Relevant'
        for n = not_Relevant'
            if(Outputs(n,1) <= Outputs(m,1))
                temp=temp+1;
            end
        end
    end

    % RankingLoss=temp/(not_Relevant_pair_size*Relevant_pair_size);
    
    AUCmicro = temp/(not_Relevant_pair_size*Relevant_pair_size);  
    
    