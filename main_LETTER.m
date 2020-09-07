function [Outputs,Pre_Labels,test_target]=main_LETTER(data,target,train_ind,test_ind,lambda1,lambda2,MaxIter)

    train_data = data(train_ind, :);
    train_target = target(:, train_ind);

    test_data = data(test_ind,:);
    test_target = target(:, test_ind);
    
    %Set the ratio parameter used by LIFT
    ratio=0.1;

    % Set the kernel type used by Libsvm
    svm.type='Linear';
    svm.para=[];

    [Outputs,Pre_Labels] = LETTER(train_data,train_target,test_data,test_target,ratio,svm, lambda1,lambda2,MaxIter);

end





