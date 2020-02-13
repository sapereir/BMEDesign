clc;close all

objective = @objFxn;

%Optimization Variables: x = [A,K,C,B,v,Q];
x0 = [1,2,1,2,1,2];

%Variable Bounds:
LB = zeros(6);
UB = 6*ones(6);

%Show Initial Objective
disp(['Initial Objective: ' num2str(objective(x0))])

%Linear Constraints
A = [];
b = [];
Aeq = [];
beq = [];

%Nonlinear constraints
nonlincon = @nonlconst;

%Optimize with fmincon
[x,fval] = fmincon(objective,x0,A,b,Aeq,beq,LB,UB,nonlincon);