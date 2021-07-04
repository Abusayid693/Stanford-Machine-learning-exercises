function [theta, J_history, theta1_history,theta2_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================

Predictions=X*theta;
Error=Predictions-y;
ComputeError=X'*Error;
theta=theta-alpha*(1/m)*ComputeError;

theta1_history(iter)=theta(1);
theta2_history(iter)=theta(2);


    % theta1_history returns the theta 1 history calculated in gradient descent interations

    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
