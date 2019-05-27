data=load('data.txt')
figure(1);clf;hold on;

% decompose in input X and output Y
n = size(data,1);
X = data(:,1:2);
Y = data(:,3);
% prepend 1s to inputs
X = [ones(n,1),X];
% Augment X to get XQuad
X(:,4)=X(:,2).*X(:,2)
X(:,5)=X(:,2).*X(:,3)
X(:,6)=X(:,3).*X(:,3)
% display the function
[a b] = meshgrid(-2:.1:2,-2:.1:2);
Xgrid = [ones(length(a(:)),1),a(:),b(:)];

l = 0;
lambda=0.001
beta=zeros(6,1)
for j=1:1:40
    l=0
    p=double.empty
    for i=1:1:length(X)
        disp(i)
        z=X(i,:)*beta
        p(i)=1/(1+exp(-z))
        l=l-(Y(i,:)*log(p(i))+(1-Y(i,:))*log(1-p(i)))
    end
    mean_nll(j)=l/n
    fprintf("Mean Neg Log Likelihood=%f",mean_nll(j))
    l=l+lambda*dot(beta,beta)
    grad_l=0;
    I=eye(6)
    p=p'
    grad_l=X'*(p-Y)+2*lambda*(I*beta)
    one=ones(length(p),1)
    W=diag(p.*(one-p))
    hess_l=X'*W*X
    beta=beta-inv(hess_l)*grad_l
end
%bar(p)
% Augment Xgrid to get Xgrid_Quad
Xgrid(:,4)=Xgrid(:,2).*Xgrid(:,2)
Xgrid(:,5)=Xgrid(:,2).*Xgrid(:,3)
Xgrid(:,6)=Xgrid(:,3).*Xgrid(:,3)

p=double.empty
%Xgrid=Xgrid(1:80,:)
ntest=length(Xgrid)
for i=1:1:ntest
     disp(i)
     z=Xgrid(i,:)*beta
     p(i)=1/(1+exp(-z))  % p(y=1|xi)

end

%bar(p)
% xlabel('index of Data point in Xgrid')
% ylabel('p(x) Blue:Training Data|Red:Test Data ')
fprintf('mean neg log likelihhod computed for the 40 iterations')
fprintf(' %f',mean_nll)
%plot3(data(:,1),data(:,2),data(:,3),'r.');
% fx = Xgrid*beta;

%  xlabel('Xgrid(1)')
%  ylabel('Xgrid(2)')
%  zlabel('Ygrid')
% % 
%  h = plot3(Xgrid(:,2),Xgrid(:,3),z);
Ygrid=Xgrid*beta
scatter3(data(:,1),data(:,2),data(:,3),'r.');
%plot(p)
xlabel('Xgrid1')
ylabel('Xgrid2')
zlabel('Ygrid')
%Ygrid=sigmoid(Ygrid)
Ygrid=reshape(Ygrid,size(a))
surface(a,b,Ygrid)
view(3)
grid on;
