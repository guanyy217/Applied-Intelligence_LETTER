
function [LabelBasedAccuracy,LabelBasedPrecision,LabelBasedRecall,LabelBasedFmeasure]=MicroLabelBasedMeasure(test_targets,predict_targets)
% syntax
%   [LabelBasedAccuracy,LabelBasedPrecision,LabelBasedRecall,LabelBasedFmeasure]=LabelBasedMeasure(test_targets,predict_targets)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   predict_targets     - L x num_test data matrix of predicted labels
%
% output
%   LabelBasedAccuracy,LabelBasedPrecision,LabelBasedRecall,LabelBasedFmeasure


    [L,~]=size(test_targets);
    test_targets=double(test_targets==1);
    predict_targets=double(predict_targets==1);
    
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    
    for i=1:L
        TP = TP + test_targets(i,:)*predict_targets(i,:)';
        FP = FP + (~test_targets(i,:).*1)*predict_targets(i,:)';
        TN = TN + (~test_targets(i,:).*1)*(~predict_targets(i,:)'.*1);
        FN = FN + test_targets(i,:)*(~predict_targets(i,:)'.*1);
    end    

    LabelBasedAccuracy  = (TP + TN)/(TP + FP + TN + FN);
    LabelBasedPrecision = TP / (TP + FP);
    LabelBasedRecall    = TP / (TP + FN);
    LabelBasedFmeasure  = 2*TP / (2*TP + FN + FP);

end