function [train_ind,test_ind] = CrossValidation(data,nfold,i)
    [n_sample,~]= size(data);
    n_test = round(n_sample/nfold);
    I = 1:n_sample;    

%     start_ind = (i-1)*n_test + 1;
%     if i==nfold
%         test_ind = start_ind:n_sample;
%     else
%         test_ind = start_ind:start_ind+n_test-1;
%     end
%     train_ind = setdiff(I,test_ind); 

    test_ind = find(mod(I, nfold) == (i-1)) ;
    train_ind = setdiff(I,test_ind);
end