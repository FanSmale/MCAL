function [entropy,richness]=multilabel_softmax(X_test,train,n,data,target,M,y,X,Y_test)
addpath('multilabel_entropy');
%%
%为了方便调用multilabel_softmax，进行一些预定义处理
unlabeledData = X_test';
[n0,M0] = size(unlabeledData);% 
unlabeledData = [ones(1,M0); unlabeledData];%每个样本都要增加一个x0=1;
numtrain=length(train);
%初始训练集
X_train=zeros(numtrain,n);
Y_train=zeros(y,numtrain);
for o=1:numtrain
    x=train(o,1);
    X_train(o,:)=data(x,:);
    Y_train(:,o)=target(:,x);
end

%信息熵初始化
entropy=zeros(1,M-numtrain);
target=target';
%丰富性初始化
richness=zeros(1,M-numtrain);

%%
%利用softmax求出样本每个标签下的信息熵和预测标签
%每个标签下执行一次softmax得到纵向的信息熵
for label=1:y
    %初始训练集转化成多个单标签集
    Y_label=Y_test(:,label);
    Y=Y_label;
    Y(find(Y==0))=2;%为了避免出现下标索引错误，将0类设为2 
    target_label=target(:,label);
    target_label(find(target_label==0))=2;
    
    %%已知代码 符合已知的softmax程序
    %第一步：预定义，为了后面调用softmax
    inputSize=n+1;%inputsize就是列的数目  这个n是列表的属性数为5
    numClasses = 2;%重新定义numClass为类别总个数   都是二分类
    lambda = 10; % Weight decay parameter衰减参数 
    theta = 0.005 * randn(inputSize*numClasses, 1);%随机产生初始的theta  randn()产生均值为0, 方差为1的正态分布的随机数  数据集的列数L2*类别的数量为行数（5*3），一列的向量
    theta = theta';%单引号表示转置，不加的时候默认以列的形式输出，加上转换成行向量输出
    inputData = X(train,: );% 训练集的名字叫inputData  令没有标签属性的数据集中的数据集名字为inputdata  
    [m1,n1] = size(inputData);%  m1:训练集样本的个数3  nl：训练集的属性列  
    inputData = inputData';% 将训练集转秩，例如3*4 -> 4*3   转置的目的是配合计算
    inputData = [ones(1,m1); inputData];%每个样本都要增加一个x0=1; 配合计算  训练集变为5*3   
    labels = target_label(train,:);%labels表示训练集的标签集 已知初始训练样本即中心点的标签
    % 第二步：利用初始训练集，训练出theta   利用softmax求出每个样本拥有每个标签的概率，用于计算信息熵  theta
    %%STEP 2.1: Implement softmaxCost
    [cost, grad] = softmax_regression_vec(theta,inputData,labels,lambda);%代价函数J（theta）
    %%STEP 2.2: Learning parameters
    options.maxIter = 7;%  最佳迭代次数 得到稳定的theta值
    softmaxModel = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options);%这个公式表示梯度下降法求解theta吗
    theta = softmaxModel.optTheta; % theta是k列，n行的矩阵 5*3  本例有3类  
    % 第三步：利用theta，对未分类实例进行分类    对测试样本求解拥有每个标签的概率，即条件概率
    predProbability1 = theta'*unlabeledData;% 得到测试样本的分类条件概率，3*5   *  5*147  theta的正负导致了predProbability1为负，exp(predProbability1)=0.000000501,就认为是0了
    predProbability2 = max(predProbability1);%根据博客上的方法，获取每一列的最大值  
    predProbability = exp(predProbability1 - predProbability2);
    sumProb = sum(predProbability);%按列相加
    [~,pred]= max(theta'*unlabeledData);%得到每个测试样本的预测分类结果，theta'*data这个矩阵如图5，某一个样本softmax最大值与这个矩阵某一列最大
    %值是等价的，因为每一列除以同一个分母和不除是一样的，并且exp（.）是增函数，所以只求里面的最大值即可。
    %acc = mean(Y_test(:) == pred(:));%判断预测标签与实际标签是否相等，并求平均，等到正确率
    %fprintf('Accuracy: %0.3f%%\n', acc * 100);    % softmax的分类准确度
    % 求信息熵
    P2 = predProbability ./ sumProb;%  3*147  ./表示点除，每个元素对应相除  定义P2为条件概率
    P3 = log(P2);   % 这里的log是以e为底的,P3=log P(y=e│x_i;θ)
    P4 = P2.*P3;   %   .*表示点乘，每个元素对应相乘
    entropy_y = -sum(P4);% 得到信息熵  entroy = -P(y=e│x_i;θ)* log 
    %将每个样本下，所有标签的预测类别进行加权平均
    %pred(find(pred==2))=0;
    richness_y=pred';%得到每个样本每个标签的丰富性量化表示
    richness_y(find(richness_y==2))=0;
    richness_y=richness_y';
    richness_label=1/y.*richness_y;
    richness=richness+richness_label;%得到每个测试样本的丰富性量化表示
    %将每个样本下，所有标签的信息熵进行加权平均
    entropy_label=1/y.*entropy_y; 
    entropy=entropy+entropy_label;  
end 
end