clc;
clear;
tic;
%biology
%load('yeast.mat');%N=103，k=14
%load('genbase.mat');%N=1185,k=27
%images
%load('scene.mat');%N=294,k=6
%load('flags.mat');%N=19,k=7
%load('corel5k.mat');%N=499,k=374
%load('corel5k-sparse.mat');
%music
%load('emotions.mat');%N=72,k=6
%load('CAL500.mat');%N=68,K=174
    %audio
    %load('birds.mat');%N=260,k=19  出错，错误使用  /  矩阵维度必须一致
    %video
    %load('mediamill.mat');%样本量很大
    %text
    %load('yahoo')
    %load('Arts_data.mat');
    %load('Arts_test.mat');
%load('medical.mat');%N=1449,k=45
%load('enron.mat');%N=1001,k=53
%load('rcv1s1.mat');%N=944,k=101
%load('rcv1s2.mat');
%load('rcv1s3.mat');
%load('rcv1s4.mat');
%load('rcv1s5.mat');
load('bibtex.mat');%N=1836,k=159    
   
    data0=[data,target'];
    [a,b]=size(data0);
    [d,a]=size(target);
    data=data0%(randperm(a,200),:);
    [M,N1]=size(data);%数据集为一个M*N的矩阵，其中每一行代表一个样本，M为数据集的行数，N为列数
    target0=data(:,N1-d+1:end);
    target=target0';
    [y,M]=size(target);
    x0=0.03;
    k1=ceil(x0*M);
    trainIdx=randperm(M,k1);
    train=data(randperm(M,k1),:);
    test=data;
    test(trainIdx,:)=[];
    train_data=train(:,1:N1-y);%从数据集中划分出train样本的数据  取出data里面train的部分
    train_target=train(:,N1-y+1:end)';%获得样本集的测试目标  ？，在本例中是实际分类情况
    test_data=test(:,1:N1-y);%test样本集  取出data里面test的部分
    test_target=test(:,N1-y+1:end)';%测试目标是目标集里的test  ？？
    toc;
    ratio=1;
    tic;
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels,num_iter]=InsDif(train_data,train_target,test_data,test_target,ratio)
    toc;