function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));



% ====================== YOUR CODE HERE ======================

i=size(z,1);
j=size(z,2);

for a=1:i;
for b=1:j;
temp=z(a,b);
sigma=1+e^(-temp);
g(a,b)=1/sigma;
end;
end;

% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================

end
