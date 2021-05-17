function [train2,q_data,q_richness]=train2_first(X,target,train,n,data,M,y,N,Data,train2)
%Based on the initial training instances, using informativeness + representativeness, 
%first select the first instance to join the training set

numtrain=length(train);
X_test=X;Y_test=target';
X_test(train,:)=[];Y_test(train,:)=[];
[M0,n0]=size(X_test);
Dists = manhattanDist(X_test,X_test);

%%
%informativeness 
[entropy,richness]=multilabel_softmax(X_test,train,n,data,target,M,y,X,Y_test);
%representativeness
[present]=multilabel_present(X_test,N);   
objectiveFun = entropy.*present;
[result,ordOF] = sort(objectiveFun,'descend');%Sort descending

%%
q=ordOF(:,1);
q_data=X_test(q,:);
q_label = Y_test(q,:);
q1 = [q_data,q_label]; 
q_richness=richness(:,q);
numtrain2=length(train2);
for i = 1:M
    if numtrain2<1
        if (Data(i,:) == q1)
            train2 = [train2;i];
            numtrain2=length(train2);
            for j=1:numtrain
                if (train(j,:)==train2)
                    train2=[];
                    numtrain2=length(train2);
                    break;
                end
            end
        end
    else
        break;
    end
end
                       
end
    
       
