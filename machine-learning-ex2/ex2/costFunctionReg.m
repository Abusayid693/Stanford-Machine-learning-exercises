function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
len=length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================

initial_hypo=X*theta;
sigmoid_cal=sigmoid(initial_hypo);


temp_1=-y'*log(sigmoid_cal);

temp_2=(1-y')*log(1-sigmoid_cal);

cost_cal=(1/m)*(temp_1-temp_2);

regularize=(lambda/(2*m)).*(theta(2:len,1).^2);


J=cost_cal+sum(regularize);
%J=regularize;
%====================== Gradient here=====================
temp_4=(1/m)*(X'*(sigmoid_cal-y));

temp_5=(lambda/m)*(theta(2:len,1));

temp_6=[0;temp_5];

grad=temp_4+temp_6;








% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
