clear all
clc
format compact
warning('off','all')

addpath(genpath('.'));
starttime = datestr(now,0);


datasetNames = ...
    ["emotions"];

nfold = 10; 
MaxIter = 10;
lambda1 = 0.001;
lambda2 = 0.001; 

for dataset = datasetNames
    % load data
    loaddir = 'data/';
    loadPath = [loaddir char(dataset) '.mat'];
    load( loadPath ); 

    target = (target>0).*1; 
    if ~isfloat(data)
        data = double(data);
    end
    
    % Normalization
    data = DataNormalization01(data); 
    
	cvResult = zeros(20,nfold); 
	ResultAll = zeros(20,2);                           

	for i = 1:nfold
        % Cross Validation
        [train_ind,test_ind] = CrossValidation(data,nfold,i);  

        % Call Method Function
        [Outputs,Pre_Labels,test_target] = main_LETTER(data,target,train_ind,test_ind,lambda1,lambda2,MaxIter);

        % Evaluation
        cvResult(:,i) = EvaluationAllMeasure(Pre_Labels,Outputs,test_target);                   
	end
	ResultAll(:,1) = ResultAll(:,1) + mean(cvResult,2);
	ResultAll(:,2) = ResultAll(:,2) + std(cvResult,1,2);   

    disp(dataset)
	disp(ResultAll');
end









