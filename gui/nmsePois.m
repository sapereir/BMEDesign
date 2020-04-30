function nmse = nmsePois(params,Q,P)
% This function computes ypred given the parameter values R, L and C.
% It then compares the ypred with y to compute the cost J.


%Model parameters
r = 0.55; %tube radius for 20Gauge Catheter
n = params(1);
l = params(2);

%Transfer function (poiseuille formula, in (kPa))
PHat = Q .* ((8*n*l)/(pi*(r^4)));

%Predicted output ypred and cost J
e = PHat - P;
nmse = var(e)/var(P);