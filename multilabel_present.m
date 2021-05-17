function [present]=multilabel_present(X_test,N)
[M0,n0]=size(X_test);
Dists1 = manhattanDist(X_test,X_test);
% No parameter probability density estimation for representativeness
dc = 0.1;
p = zeros(1, M0);
for i=1 : M0 - 1 
    for j = i+1 : M0
        p(i)=p(i) + exp(-(Dists1(i,j)/sqrt(2)/dc)*(Dists1(i,j)/sqrt(2)/dc));%p（x）
        p(j)=p(j) + exp(-(Dists1(i,j)/sqrt(2)/dc)*(Dists1(i,j)/sqrt(2)/dc));%相当于求和
    end
end
present = p/sqrt(2*pi*N);

end