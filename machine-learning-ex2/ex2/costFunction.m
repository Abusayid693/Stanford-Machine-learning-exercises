function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================

%=========COST HERE===============

initial_hypo=X*theta;
sigmoid_cal=sigmoid(initial_hypo);

temp_1=-y'*log(sigmoid_cal);

temp_2=(1-y')*log(1-sigmoid_cal);

cost_cal=temp_1-temp_2;
J=(1/m).*cost_cal;

%========GRADIENT HERE============


temp_3=X'*(sigmoid_cal-y);
grad=(1/m)*temp_3;







% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
