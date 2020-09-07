function AUCmacro=AUC_macro(Outputs,test_target)
%Computing the hamming loss
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance]=size(Outputs);
%     temp_Outputs=[];
%     temp_test_target=[];
%     for i=1:num_instance
%         temp=test_target(:,i);
%         if((sum(temp)~=num_class)&&(sum(temp)~=-num_class))
%             temp_Outputs=[temp_Outputs,Outputs(:,i)];
%             temp_test_target=[temp_test_target,temp];
%         end
%     end
%     Outputs=temp_Outputs;
%     test_target=temp_test_target;     
%     [num_class,num_instance]=size(Outputs);
    
    
    AUCmacro = 0;
    for j=1:num_class
        label_true = length(find(test_target(j,:)==1));
        label_false = length(find(test_target(j,:)~=1));
        label_true_idx = find(test_target(j,:)==1);
        label_false_idx = find(test_target(j,:)~=1);

        macro_auc_j = 0;
        for i = label_true_idx
           auc_j = length(find(Outputs(j,label_false_idx) <= Outputs(j,i)));
           macro_auc_j = macro_auc_j + auc_j/(label_true*label_false);
        end
        AUCmacro = AUCmacro + macro_auc_j;
    end    
       
    AUCmacro = AUCmacro/num_class;
    % AUCmacro = 1-RankingLoss; 


    
    