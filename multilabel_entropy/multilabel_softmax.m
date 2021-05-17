function [entropy,richness]=multilabel_softmax(X_test,train,n,data,target,M,y,X,Y_test)
addpath('multilabel_entropy');
%%
%Ϊ�˷������multilabel_softmax������һЩԤ���崦��
unlabeledData = X_test';
[n0,M0] = size(unlabeledData);% 
unlabeledData = [ones(1,M0); unlabeledData];%ÿ��������Ҫ����һ��x0=1;
numtrain=length(train);
%��ʼѵ����
X_train=zeros(numtrain,n);
Y_train=zeros(y,numtrain);
for o=1:numtrain
    x=train(o,1);
    X_train(o,:)=data(x,:);
    Y_train(:,o)=target(:,x);
end

%��Ϣ�س�ʼ��
entropy=zeros(1,M-numtrain);
target=target';
%�ḻ�Գ�ʼ��
richness=zeros(1,M-numtrain);

%%
%����softmax�������ÿ����ǩ�µ���Ϣ�غ�Ԥ���ǩ
%ÿ����ǩ��ִ��һ��softmax�õ��������Ϣ��
for label=1:y
    %��ʼѵ����ת���ɶ������ǩ��
    Y_label=Y_test(:,label);
    Y=Y_label;
    Y(find(Y==0))=2;%Ϊ�˱�������±��������󣬽�0����Ϊ2 
    target_label=target(:,label);
    target_label(find(target_label==0))=2;
    
    %%��֪���� ������֪��softmax����
    %��һ����Ԥ���壬Ϊ�˺������softmax
    inputSize=n+1;%inputsize�����е���Ŀ  ���n���б��������Ϊ5
    numClasses = 2;%���¶���numClassΪ����ܸ���   ���Ƕ�����
    lambda = 10; % Weight decay parameter˥������ 
    theta = 0.005 * randn(inputSize*numClasses, 1);%���������ʼ��theta  randn()������ֵΪ0, ����Ϊ1����̬�ֲ��������  ���ݼ�������L2*��������Ϊ������5*3����һ�е�����
    theta = theta';%�����ű�ʾת�ã����ӵ�ʱ��Ĭ�����е���ʽ���������ת�������������
    inputData = X(train,: );% ѵ���������ֽ�inputData  ��û�б�ǩ���Ե����ݼ��е����ݼ�����Ϊinputdata  
    [m1,n1] = size(inputData);%  m1:ѵ���������ĸ���3  nl��ѵ������������  
    inputData = inputData';% ��ѵ����ת�ȣ�����3*4 -> 4*3   ת�õ�Ŀ������ϼ���
    inputData = [ones(1,m1); inputData];%ÿ��������Ҫ����һ��x0=1; ��ϼ���  ѵ������Ϊ5*3   
    labels = target_label(train,:);%labels��ʾѵ�����ı�ǩ�� ��֪��ʼѵ�����������ĵ�ı�ǩ
    % �ڶ��������ó�ʼѵ������ѵ����theta   ����softmax���ÿ������ӵ��ÿ����ǩ�ĸ��ʣ����ڼ�����Ϣ��  theta
    %%STEP 2.1: Implement softmaxCost
    [cost, grad] = softmax_regression_vec(theta,inputData,labels,lambda);%���ۺ���J��theta��
    %%STEP 2.2: Learning parameters
    options.maxIter = 7;%  ��ѵ������� �õ��ȶ���thetaֵ
    softmaxModel = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options);%�����ʽ��ʾ�ݶ��½������theta��
    theta = softmaxModel.optTheta; % theta��k�У�n�еľ��� 5*3  ������3��  
    % ������������theta����δ����ʵ�����з���    �Բ����������ӵ��ÿ����ǩ�ĸ��ʣ�����������
    predProbability1 = theta'*unlabeledData;% �õ����������ķ����������ʣ�3*5   *  5*147  theta������������predProbability1Ϊ����exp(predProbability1)=0.000000501,����Ϊ��0��
    predProbability2 = max(predProbability1);%���ݲ����ϵķ�������ȡÿһ�е����ֵ  
    predProbability = exp(predProbability1 - predProbability2);
    sumProb = sum(predProbability);%�������
    [~,pred]= max(theta'*unlabeledData);%�õ�ÿ������������Ԥ���������theta'*data���������ͼ5��ĳһ������softmax���ֵ���������ĳһ�����
    %ֵ�ǵȼ۵ģ���Ϊÿһ�г���ͬһ����ĸ�Ͳ�����һ���ģ�����exp��.����������������ֻ����������ֵ���ɡ�
    %acc = mean(Y_test(:) == pred(:));%�ж�Ԥ���ǩ��ʵ�ʱ�ǩ�Ƿ���ȣ�����ƽ�����ȵ���ȷ��
    %fprintf('Accuracy: %0.3f%%\n', acc * 100);    % softmax�ķ���׼ȷ��
    % ����Ϣ��
    P2 = predProbability ./ sumProb;%  3*147  ./��ʾ�����ÿ��Ԫ�ض�Ӧ���  ����P2Ϊ��������
    P3 = log(P2);   % �����log����eΪ�׵�,P3=log P(y=e��x_i;��)
    P4 = P2.*P3;   %   .*��ʾ��ˣ�ÿ��Ԫ�ض�Ӧ���
    entropy_y = -sum(P4);% �õ���Ϣ��  entroy = -P(y=e��x_i;��)* log 
    %��ÿ�������£����б�ǩ��Ԥ�������м�Ȩƽ��
    %pred(find(pred==2))=0;
    richness_y=pred';%�õ�ÿ������ÿ����ǩ�ķḻ��������ʾ
    richness_y(find(richness_y==2))=0;
    richness_y=richness_y';
    richness_label=1/y.*richness_y;
    richness=richness+richness_label;%�õ�ÿ�����������ķḻ��������ʾ
    %��ÿ�������£����б�ǩ����Ϣ�ؽ��м�Ȩƽ��
    entropy_label=1/y.*entropy_y; 
    entropy=entropy+entropy_label;  
end 
end