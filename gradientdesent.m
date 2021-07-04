


function R= gradientdesent(theta,Y,X,u);
 m=size(X,1);
 predictions=X*theta;
 error=predictions-Y;
 sumerror=(u*(1/m)).*X'*error;
 R=theta-sumerror;
 end;
