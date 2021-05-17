function [cl,centers,rho]=density(X,k)

[n, d] = size(X);
Dists = manhattanDist(X, X); % Distance measure
maxDist = max(max(Dists));
dc=0.1;
dc = dc * maxDist;
rho = zeros(n, 1);
% Compute rho, where rho >= 0
% for i = 1:n
%     rho(i) = sum(Dists(i, :) < dc) - 1; 
% end

for i=1:n-1
    for j=i+1:n
        rho(i)=rho(i)+exp(-(Dists(i,j)/dc)*(Dists(i,j)/dc));
        rho(j)=rho(j)+exp(-(Dists(i,j)/dc)*(Dists(i,j)/dc));
    end
end

delta = zeros(n, 1);
%master = zeros(n, 1);
master = -ones(n, 1);
[~, ordrho] = sort(rho, 'descend');
delta(ordrho(1)) = maxDist;
for i = 2:n
    delta(ordrho(i)) = maxDist;
    for j = 1:i-1
        if Dists(ordrho(i), ordrho(j)) < delta(ordrho(i))
            delta(ordrho(i)) = Dists(ordrho(i), ordrho(j));
            master(ordrho(i)) = ordrho(j);
        end
    end
end
gamma = rho .* delta;
[~, desInd] = sort(gamma, 'descend');

% Compute centers  
tblock = k;
centers = desInd(1:tblock, 1);
% cluster with centers
cl = -ones(n, 1);
cl(centers) = 1:tblock;
    
for i = 1:n
    if cl(ordrho(i)) == -1
       cl(ordrho(i)) = cl(master(ordrho(i)));
    end
end

    
