function [RankingLoss,OneError,Coverage,Average_Precision]=multilabel_InsDif(train,N,y,Data)
addpath('multilabel_InsDif');
addpath('evaluation');
numtrain=length(train);
trainset=zeros(numtrain,N)
for w=1:numtrain
    s=train(w,1);
    trainset(w,:)=Data(s,:);
end
train_data=trainset(:,1:N-y);
train_target0=trainset(:,N-y+1:end);
train_target=train_target0';
testset=Data;testset(train,:)=[];
test_data=testset(:,1:N-y);
test_target0=testset(:,N-y+1:end);
test_target=test_target0';


%%
ratio=1;
[RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels,num_iter]=INSDIF(train_data,train_target,test_data,test_target,ratio)
   
end