
function Accuracy=AccuracyEvaluation(Pre_Labels,test_target)
%% Evaluate the classification accuracy for single-label learning
% Input:
%       predict_target: 1 X N vector, predicted result by classifier
%       test_target   £º1 X N vector, the ground truth label set
%
% Output:
%       Accuracy      : classifer's accuracy

%     num_test=size(test_target,2);
%     correctones=(predict_target==test_target);
%     Accuracy=sum(correctones)/num_test;
    
    [num_class,num_instance]=size(Pre_Labels);
    total=0;
    
    
    for i=1:num_instance
        numerator=Pre_Labels(:,i)'*test_target(:,i);
        denominator=sum(or(Pre_Labels(i,:),Pre_Labels(i,:)));

        if denominator ~=0
            total=total+numerator/denominator;
        end
    end
    Accuracy=total/num_instance;    

end