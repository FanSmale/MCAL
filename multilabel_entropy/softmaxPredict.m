function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  %theta是k列，n行的矩阵
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
[~,pred]= max(theta'*data);%theta'*data这个矩阵如图5，某一个样本softmax最大值与这个矩阵某一列最大
%值是等价的，因为每一列除以同一个分母和不除是一样的，并且exp（.）是增函数，所以只求里面的最大值即可。







% ---------------------------------------------------------------------

end

