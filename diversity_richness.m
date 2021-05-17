function [train]=diversity_richness(M,numtrain,x,train1,train2,X,target,n,data,y,N,Dists_threshold,q_data_new,q_richness_new,Data)
numtrain=length(train1)+length(train2);
for a=1:M
    if numtrain<x+1
        train=[train1;train2];
        numtrain=length(train);
        X_test=X;Y_test=target';
        X_test(train,:)=[];Y_test(train,:)=[];

        %informativeness 
        [entropy,richness]=multilabel_softmax(X_test,train,n,data,target,M,y,X,Y_test);
        Meanrichness_new=mean(richness);
        %representativeness
        [present]=multilabel_present(X_test,N);   
        objectiveFun = entropy.*present;
        [result,ordOF] = sort(objectiveFun,'descend'); 
        %Ensure that a point is selected in each round
        for b=1:(M-numtrain)
            new=ordOF(b);
            richness_new=richness(new);
            %select new training instance unless diversity+richness
            %constraints be met
           Dists_g = sum(abs((X_test(new,:) - q_data_new)));
           if (Dists_g>Dists_threshold)&(Meanrichness_new<=richness_new)
               break;
           end
        end
        
        %Reassignment definition to add new cycle
        q_data_new = X_test(new,:);
        q_label_new = Y_test(new,:);
        q_new = [q_data_new,q_label_new]; 
        q_richness_new=richness(:,new);
     %Determine the sequence number of the training instance to be selected and add it to the training set   
    train2_0=[];    
    numtrain2_0=length(train2_0);
    for i = 1:M
        if numtrain2_0<1
            if (Data(i,:) == q_new)
                train2_0 = [train2_0;i];
                numtrain2_0=length(train2_0);
                for j=1:numtrain
                    if (train(j,:)==train2_0)
                        train2_0=[];
                        numtrain2_0=length(train2_0);
                        break;
                    end
                end
            end
        else
            break;
        end
    end
        
        train2=[train2;train2_0];
        numtrain=length(train1)+length(train2);
    else
        break     
    end 
    end

end