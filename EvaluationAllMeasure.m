function ResultAll = EvaluationAllMeasure(Pre_Labels,Outputs,test_target)
% evluation for MLC algorithms, there are fifteen evaluation metrics
% 
% syntax
%   ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   Pre_Labels          - L x num_test data matrix of predicted labels
%   Outputs             - L x num_test data matrix of scores
%
% output
%     ResultAll=zeros(20,1); 
%     ResultAll(1,1)=SubsetAccuracy;
%     ResultAll(2,1)=HammingLoss;
%     ResultAll(3,1)=ExampleBasedAccuracy; 
%     ResultAll(4,1)=ExampleBasedPrecision; 
%     ResultAll(5,1)=ExampleBasedRecall; 
%     ResultAll(6,1)=ExampleBasedFmeasure;
% 
%     ResultAll(7,1)=OneError;
%     ResultAll(8,1)=Coverage;
%     ResultAll(9,1)=RankingLoss;
%     ResultAll(10,1)=Average_Precision;   
%     
%     ResultAll(11,1)=MacLabelBasedAccuracy; 
%     ResultAll(12,1)=MacLabelBasedPrecision;
%     ResultAll(13,1)=MacLabelBasedRecall; 
%     ResultAll(14,1)=MacLabelBasedFmeasure; 
% 
%     ResultAll(15,1)=MicLabelBasedAccuracy; 
%     ResultAll(16,1)=MicLabelBasedPrecision;
%     ResultAll(17,1)=MicLabelBasedRecall; 
%     ResultAll(18,1)=MicLabelBasedFmeasure;

%     ResultAll(19,1)=AUCmacro; 
%     ResultAll(20,1)=AUCmicro; 

    Pre_Labels(Pre_Labels==0) = -1;
    test_target(test_target==0) = -1;

    % 01 SubsetAccuracy
    SubsetAccuracy=SubsetAccuracyEvaluation(test_target,Pre_Labels);
    
    % 02 HammingLoss
    HammingLoss=Hamming_loss(Pre_Labels,test_target);
    
    % 03 04 05 06 Accuracy Precision Recall Fmeasure
    [ExampleBasedAccuracy,ExampleBasedPrecision,ExampleBasedRecall,ExampleBasedFmeasure]=ExampleBasedMeasure(test_target,Pre_Labels);
    
    % 07 OneErro
    OneError=One_error(Outputs,test_target);
    
    % 08 Coverage
    Coverage=coverage(Outputs,test_target);
    
    % 09 RankingLoss
    RankingLoss=Ranking_loss(Outputs,test_target);
    
    % 10 Average_Precision
    Average_Precision=Average_precision(Outputs,test_target);
    
    % 11-14 MacLabelBasedMeasure
    [MacLabelBasedAccuracy,MacLabelBasedPrecision,MacLabelBasedRecall,MacLabelBasedFmeasure]=MacroLabelBasedMeasure(test_target,Pre_Labels);
    
    % 15-18 MicLabelBasedMeasure
    [MicLabelBasedAccuracy,MicLabelBasedPrecision,MicLabelBasedRecall,MicLabelBasedFmeasure]=MicroLabelBasedMeasure(test_target,Pre_Labels);   
    
    % 19 AUCmacro
    AUCmacro = AUC_macro(Outputs,test_target);
    
    % 20 AUCmicro
    AUCmicro = AUC_micro(Outputs,test_target);
    
    ResultAll=zeros(20,1); 
    ResultAll(1,1)=SubsetAccuracy;
    ResultAll(2,1)=HammingLoss;
    ResultAll(3,1)=ExampleBasedAccuracy; 
    ResultAll(4,1)=ExampleBasedPrecision; 
    ResultAll(5,1)=ExampleBasedRecall; 
    ResultAll(6,1)=ExampleBasedFmeasure;

    ResultAll(7,1)=OneError;
    ResultAll(8,1)=Coverage;
    ResultAll(9,1)=RankingLoss;
    ResultAll(10,1)=Average_Precision;   
    
    ResultAll(11,1)=MacLabelBasedAccuracy; 
    ResultAll(12,1)=MacLabelBasedPrecision;
    ResultAll(13,1)=MacLabelBasedRecall; 
    ResultAll(14,1)=MacLabelBasedFmeasure; 

    ResultAll(15,1)=MicLabelBasedAccuracy; 
    ResultAll(16,1)=MicLabelBasedPrecision;
    ResultAll(17,1)=MicLabelBasedRecall; 
    ResultAll(18,1)=MicLabelBasedFmeasure;

    ResultAll(19,1)=AUCmacro; 
    ResultAll(20,1)=AUCmicro;    
 
end