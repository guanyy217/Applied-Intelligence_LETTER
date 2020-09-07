function [data] = DataNormalization01(data)
    data = double(data);

%     normdata = data - mean(data);
%     normdata = normdata ./ std(data);   
%     normdata(find(isnan(normdata)))=0; 
%     data = normdata;
    
%     [normdata,~] = mapminmax(data);
    normdata = data - min(data);
    normdata = normdata ./ max(normdata);            
    normdata(find(isnan(normdata)))=0; 

    data = normdata;
%     data = 2.*normdata-1;
end