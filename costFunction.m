

function [jVal, gradient] = costFunction(theta)

%X is training set data is matrix format
%Y is training set output in vectoR FORMAT

% ======Call fminunc function with ititial theta vector zero===========

  X=[1 2;1 2.2];

%======== our hypothesis only contain theta 0 and theta 1 i.e h(x)=theta0 * x0 + theta1 * x1=======

  Y=[20;20.2];

  a=(X*theta-Y).^2;
  m=length(X);
  jVal = (1/(2*m))*sum(a);
  gradient = zeros(2, 1);
  gradient(1) = (1/m)*sum((X*theta-Y));
  gradient(2) = (1/m)*sum((X*theta -Y).*X(:,2));

end