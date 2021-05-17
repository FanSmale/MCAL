clc;
clear;
Result=zeros(10,4);
for i=1:10
    [result]=multilabel()
    Result(i,:)=result;
end

RESULT=zeros(2,4);
for i=1:4
    X=Result(:,i);
    RESULT(1,i)=mean(X);%mean
    RESULT(2,i)=sum((X(:,1)-mean(X)).^2)/length(X);%var
end





