function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  %theta��k�У�n�еľ���
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
[~,pred]= max(theta'*data);%theta'*data���������ͼ5��ĳһ������softmax���ֵ���������ĳһ�����
%ֵ�ǵȼ۵ģ���Ϊÿһ�г���ͬһ����ĸ�Ͳ�����һ���ģ�����exp��.����������������ֻ����������ֵ���ɡ�







% ---------------------------------------------------------------------

end

