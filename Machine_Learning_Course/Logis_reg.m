
data=load('data.txt')
figure(1);clf;hold on;

% decompose in input X and output Y
n = size(data,1);
X = data(:,1:2);
Y = data(:,3);
% prepend 1s to inputs
X = [ones(n,1),X];
% compute optimal beta

% display the function
[a b] = meshgrid(-2:.1:2,-2:.1:2);
Xgrid = [ones(length(a(:)),1),a(:),b(:)];

l = 0;

lambda=0.001 %0.0001
%for lambda=0.0001:1000000
beta=zeros(3,1)

for j=1:1:20 % or you can use a tolerance condition
    l=0
    p=double.empty
    for i=1:1:n   %n=size(data,1) or n=length(X)
        disp(i)
        z=X(i,:)*beta
        p(i)=1/(1+exp(-z))
        l=l-(Y(i,:)*log(p(i))+(1-Y(i,:))*log(1-p(i)))
    end
    mean_nll(j)=l/n
    fprintf("Mean Neg Log Likelihood=%f",mean_nll(j))
    l=l+lambda*dot(beta,beta) % L(beta) = L(beta) + lambda*(beta^2)
    grad_l=0;
    I=eye(3)
    p=p'
    grad_l=X'*(p-Y)+2*lambda*(I*beta)
    one=ones(length(p),1)
    W=diag(p.*(one-p))
    hess_l=X'*W*X
    beta=beta-inv(hess_l)*grad_l
    %lambda=lambda*10
end
%end
%plot(p)
p=double.empty
%Xgrid=Xgrid(1:80,:)
ntest=length(Xgrid)
for i=1:1:ntest
     disp(i)
     z=Xgrid(i,:)*beta
     p(i)=1/(1+exp(-z))  % p(y=1|xi)

end
%scatter3 ,reshape ygrid with size(a) , surface plot a,b,ygrid, plot
%probabilities as sigmmoid of (ygrid) +reshape
%scatter3(X(:,1),X(:,2),X(:,3),'r.');
Ygrid=Xgrid*beta
scatter3(data(:,1),data(:,2),data(:,3),'r.');
%plot(p)
xlabel('Xgrid1')
ylabel('Xgrid2')
zlabel('Ygrid')
%Ygrid=sigmoid(Ygrid)
Ygrid=reshape(Ygrid,size(a))
surface(a,b,Ygrid)
fprintf('mean neg log likelihhod computed for the 20 iterations')
fprintf(' %f',mean_nll)
view(3);
grid on;