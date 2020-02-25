clc;close all;clear all;

%Import the data
s = 'C:\Users\illum\Documents\School\Spring 20\BME Design\BME Design Math Model Attempt 3\MathModel\GageModel\RawData.txt';
RawData = importRawData(s);

%Want numDataPoints data points
numDataPoints = 1000;
contrastType = 'C30';
salineType = 'S90';
flowRate = 6;
PartialData = parsePressureData(RawData,20,flowRate,contrastType,salineType,numDataPoints);

exampleTimes = squeeze(PartialData(:,2,:));
examplePressures = squeeze(PartialData(:,3,:));

%Still, some rows are entirely zeros, so I need to loop thorugh the final
%Example data set and remove the rows that are all zeros
% for i = 1:length(examplePressures)
%     if 1 && all(examplePressures(i) == 0)
%         examplePressures(i) = [];
%     end
% end

objective = @(x) objFxn(x,nonzeros(exampleTimes),nonzeros(examplePressures));

%Optimization Variables: x = [K,B,v,Q];
x0 = [1,1,1,1];

%Variable Bounds:
LB = zeros(4);
UB = [10000,10000,10000,10000];

%Linear Constraints
A = [];
b = [];
Aeq = [];
beq = [];

%Nonlinear constraints
nonlincon = @nonlconst;

%Optimize with fmincon
[x,fval] = fmincon(objective,x0,A,b,Aeq,beq,LB,UB,nonlincon);

%Plot of Logistics function fit to data for a given gauge
Kf = x(1);
Bf = x(2);
vf = x(3);
Qf = x(4);

tf = linspace(0,20,100); %seconds
curveFin = (Kf)./(1 + Qf.*exp(-Bf.*tf)).^(1./vf);

%Compare with some random pressure curves from the data set
r = randi([1,numDataPoints],3,1); %3x1 column vector of random indicies in our data set
r1 = r(1,1);
r2 = r(2,1);
r3 = r(3,1);

times1 = nonzeros(exampleTimes(r1,:));
pressures1 = nonzeros(examplePressures(r1,:));
times2 = nonzeros(exampleTimes(r2,:));
pressures2 = nonzeros(examplePressures(r2,:));
times3 = nonzeros(exampleTimes(r3,:));
pressures3 = nonzeros(examplePressures(r3,:));

figure;
plot(tf,curveFin);
hold on
plot(times1,pressures1);
hold on
plot(times2,pressures2);
hold on
plot(times3,pressures3);
hold on
title('Fit Logistics Curve');
xlabel('Time (s)');
ylabel('Pressure (kPa)');