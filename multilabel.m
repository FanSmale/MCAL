function [result]=multilabel()
clc;
clear;
%%
%informativeness:multilabel_softmax
%representativeness:multilabel_present
%The first candidate training instance:train2_first
%diversity+richness:diversity_richness
%labels prediction:multilabel_INSDIF

addpath('benchmark_data');addpath('Yahoo_data');
addpath('multilabel_entropy');
addpath('multilabel_InsDif');
addpath('evaluation');

%%
load('Flags.mat');thre=0.01;
Data=[data,target'];
[M,n]=size(data);
[y,M]=size(target);
N=n+y;
X=data;
Dists = manhattanDist(X, X);

%%
%Select the initial training instances
r=0.05;
k=ceil( M*0.5*r);[cl,center,rho]=density(X,k);train1=center;
x =ceil( M*r);
train2=[];
train=[train1,train2];
numtrain=length(train);

%%
%Select the first candidate instance
[train2,q_data,q_richness]=train2_first(X,target,train,n,data,M,y,N,Data,train2);
q_data_new=q_data;
q_richness_new=q_richness;

%%
%diversity+richness constraints screen new instances
MaxDists=max( mean(Dists,2) );
Dists_threshold = thre * MaxDists;%diversity threshold
[train]=diversity_richness(M,numtrain,x,train1,train2,X,target,n,data,y,N,Dists_threshold,q_data_new,q_richness_new,Data);
    
%%
%INSDIF predict label sets
[RankingLoss,OneError,Coverage,Average_Precision]=multilabel_INSDIF(train,N,y,Data);
result=zeros(1,4);
result(1,:)=[Average_Precision;Coverage;OneError;RankingLoss];

end

