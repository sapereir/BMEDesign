function nmse = nmsePois3(params,Q,P,t)
% This function computes PHat given the parameter values k and tau.
% It then compares the PHat with P to compute the cost J.

%Model parameters
k = params(1);
tau = params(2);

%Transfer function
num = k;
den = [tau, 1];
H = tf(num,den);

%Predicted output ypred and cost J
PHat = lsim(H,Q,t);
e = P - PHat';
nmse = var(e)./var(P);