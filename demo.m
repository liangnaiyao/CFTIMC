clear all;
addpath('./dataset');

%% =====================  setting =====================
options.dataStr='handwritten1';  % 'handwritten1' | 'handwritten2'
load(char(options.dataStr));
options.sigma=1; 
options.gamma=1; 
options.lambda=10;
options.miss=0.1; % missing rate
if strcmp(options.dataStr,'handwritten1')
    normalization=1;
elseif strcmp(options.dataStr,'handwritten2')
    normalization=2;
else
    normali=0;
end
%% =====================  run =====================
for fold_i=1:15  
    options.fold_i=fold_i;
    [X_part,options]=gen_incom_data(X,options);   
	%% =====  Data normalization ======
    X_part=data_normalization(X_part,normalization); 
    [S,V,Q] = CFTIMC(X_part,label,options);
	%% =====================  result =====================
    printResult(V, label', numel(unique(label)), 1);
end 


%%
function data=data_normalization(data,normalization)
if normalization==0   
elseif normalization==1    
    for v = 1:length(data)
     data{v} = data{v}*diag(sparse(1./sqrt(sum(data{v}.^2))));
    end  
 elseif normalization==2  
    num=size(data{1},2);
    v=length(data);
    for i = 1 :v  
        for  j = 1:num 
            data{i}(:,j) = ( data{i}(:,j)- mean( data{i}(:,j) ) ) / std( data{i}(:,j) ) ; 
        end 
    end
end
end

function [data,parameter]=gen_incom_data(data,parameter)
numOfDatas=size(data{1},2);
data0=data; 
nv=length(data);
 
percentDel=parameter.miss;
Datafold = [char(parameter.dataStr),'_miss_',num2str(percentDel),'.mat']; 
load(Datafold);  
ind_folds = folds{parameter.fold_i};
X_0=data0;
clear data;
WI = eye(numOfDatas); 
for iv = 1:nv  
    ind_0{iv} = find(ind_folds(:,iv) == 0); 
    ind_1{iv} = find(ind_folds(:,iv) == 1); 
    data{iv} = data0{iv}(:,ind_1{iv});   
    X_0{iv}(:,ind_0{iv}) = 0; 
    W1{iv} = WI(:,ind_1{iv});   
    W0{iv} = WI(:,ind_0{iv});  
end
parameter.W1=W1;
parameter.W0=W0;
end