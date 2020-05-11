%% 42402 Mathematical Model for Ricky, Gage, and Anusha

%% Load Data

time = [0.000, 0.355, 0.591, 0.877, 1.230, 1.528, 1.878, 2.174, 2.569, 2.880, 3.133, 3.431, 3.731, 4.031, 4.327, 4.599, 4.853, 5.103, 5.356, 5.637, 5.968, 6.279, 6.628, 6.985, 7.268, 7.528, 7.872, 8.195, 8.458, 8.780, 9.167, 9.518, 9.829, 10.090, 10.347, 10.590, 10.840, 11.118, 11.366, 11.640, 11.863, 12.114, 12.352, 12.608, 12.832, 13.083, 13.356, 13.601, 13.857, 14.106, 14.448, 14.690, 14.985, 15.317, 15.610, 15.869, 16.124, 16.454, 16.767, 17.077, 17.324, 17.612, 17.875, 18.174, 18.523, 18.861, 19.124, 19.476, 19.714, 19.984, 20.244, 20.482, 20.736, 21.067, 21.367, 21.676, 21.973, 22.329];
pressure = [0, 0, 69, 145, 234, 296, 496, 600, 814, 903, 993, 1076, 1165, 1207, 1276, 1282, 1317, 1317, 1372, 1379, 1393, 1400, 1413, 1434, 1400, 1427, 1448, 1420, 1427, 1400, 1434, 1434, 1434, 1420, 1413, 1420, 1407, 1420, 1393, 1441, 1372, 1393, 1365, 1420, 1372, 1365, 1358, 1393, 1365, 1365, 1331, 1358, 1331, 1358, 1317, 1331, 1358, 1262, 1069, 834, 910, 924, 889, 903, 834, 779, 752, 696, 676, 655, 621, 607, 621, 593, 586, 579, 517, 352];
flow = [0.00, 0.12, 4.13, 6.10, 5.99, 5.94, 6.02, 5.94, 5.99, 6.10, 5.86, 6.03, 6.00, 5.94, 6.01, 6.06, 5.93, 6.01, 6.04, 5.98, 5.97, 5.94, 6.03, 5.99, 5.99, 5.99, 5.99, 6.06, 5.88, 6.07, 6.00, 5.90, 6.00, 6.01, 5.92, 6.07, 5.93, 6.10, 5.93, 5.97, 6.03, 5.90, 5.96, 6.05, 5.98, 6.03, 5.97, 6.01, 5.97, 5.96, 6.04, 6.01, 5.95, 5.97, 5.98, 6.00, 5.99, 5.33, 2.26, 0.62, 4.84, 6.01, 5.98, 6.00, 5.98, 5.98, 6.06, 5.97, 6.01, 5.98, 6.02, 5.98, 5.97, 6.01, 5.92, 6.10, 5.08, 0.98];

% Visualize Data
figure
subplot(2,1,1); 
    plot(time, pressure);
    xline(5);
    xline(15);
    xline(17.077);
    xlabel('time(sec)');
    ylabel('pressure(kPa)');
    title('Catheter Flow and Pressure Data');
    
subplot(2,1,2); 
    plot(time, flow);
    xline(5);
    xline(15);
    xline(17.077);
    xlabel('time(sec)');
    ylabel('flow rate(mL/s)');

new_time = (time < 16);    % Focus on contrast phase to increase accuracy
flow = flow(new_time);
pressure = pressure(new_time);
time = time(new_time);
% Visualize data subset
figure
subplot(2,1,1); 
    plot(time, pressure);
    xlabel('time(sec)');
    ylabel('pressure(kPa)');
    title('New Catheter Flow and Pressure Data');
subplot(2,1,2); 
    plot(time, flow);
    xlabel('time(sec)');
    ylabel('flow rate(mL/s)');


%% Cross Correlate Flow and Pressure

[cross_cor, lag] = xcorr((flow - mean(flow)), ...
                        (pressure - mean(pressure)), 'coeff');

figure
plot(lag, cross_cor);
title('Cross Correlation of Pressure vs. Flow Rate');
xlabel('Lag');
ylabel('Correlation Coefficient');

% Graph shows no lag and positive relationship (as expected)

%% Redefine New Transfer Function From Frequency Analysis

Q = flow; % in mL/sec
P = pressure; % in kPa

fs = 100;

%An arbitrary filter (impulse response h(t))

N = length(Q);
nfft = 2^nextpow2(N); %number of points in spectrum

%FFT of input
X = fft(Q - mean(Q),nfft);
Xmag = abs(fftshift(X));
Xpha = angle(fftshift(X));

%FFT of output
Y = fft(P - mean(P),nfft);
Ymag = abs(fftshift(Y));
Ypha = angle(fftshift(Y));

%Frequency response
H = Y ./ X;
Hmag = abs(fftshift(H));
Hpha = angle(fftshift(H));

%Impulse response
hhat = ifft(H); %take inverse FT to get back to time domain

%Compare h: true vs. estimated
figure; plot(hhat);
title('Frequency Domain Analysis: Estimated h(t)');
xlabel('Time (sec)');

% Testing on new data set.
time2 = [0.000, 0.319, 0.592, 0.926, 1.185, 1.471, 1.785, 2.106, 2.412, 2.671, 2.967, 3.166, 3.403, 3.726, 4.063, 4.364, 4.670, 4.968, 5.276, 5.497, 5.819, 6.145, 6.470, 6.776, 7.067, 7.416, 7.658, 7.976, 8.334, 8.673, 8.917, 9.171, 9.419, 9.664, 9.910, 10.151, 10.469, 10.761, 11.118, 11.379, 11.670, 11.948, 12.197, 12.542, 12.789, 13.074, 13.409, 13.728, 13.971, 14.248, 14.575, 14.854, 15.166, 15.517, 15.761, 16.117, 16.390, 16.770, 17.067, 17.417, 17.669, 17.914, 18.271, 18.565, 18.868, 19.170, 19.517, 19.759, 20.070, 20.418, 20.743, 21.069, 21.307, 21.663, 22.010, 22.214, 22.530, 22.868, 23.175, 23.445, 23.749, 24.009, 24.266, 24.508, 24.763, 24.995, 25.312, 25.572, 25.962, 26.161, 26.470, 26.774, 27.121, 27.428, 27.698, 28.014, 28.353, 28.624, 29.014, 29.198, 29.518, 29.819, 30.125, 30.424, 30.768, 31.016, 31.295, 31.622, 31.916, 32.178, 32.531];
pressure2 = [0, 0, 0, 0, 0, 14, 48, 117, 138, 159, 179, 221, 241, 276, 317, 324, 338, 379, 400, 407, 421, 427, 427, 448, 462, 462, 476, 469, 469, 490, 490, 490, 496, 503, 490, 483, 490, 496, 496, 503, 490, 483, 483, 496, 490, 503, 503, 490, 496, 503, 496, 503, 503, 490, 483, 503, 496, 510, 496, 476, 496, 496, 496, 483, 496, 483, 483, 490, 490, 483, 490, 469, 476, 483, 476, 476, 462, 455, 448, 448, 324, 221, 262, 283, 303, 303, 283, 255, 179, 152, 131, 110, 97, 83, 76, 76, 76, 62, 55, 55, 55, 55, 55, 48, 48, 55, 48, 48, 55, 48, 0];
flow2 = [0.00, 0.00, 2.58, 2.98, 2.98, 3.02, 2.99, 3.00, 2.94, 3.06, 2.96, 3.00, 3.02, 2.96, 2.98, 3.00, 2.98, 3.03, 2.96, 3.00, 3.00, 3.00, 2.98, 3.00, 2.98, 3.00, 2.98, 3.02, 3.00, 3.00, 2.95, 3.05, 2.96, 2.99, 3.03, 2.97, 3.03, 2.98, 3.00, 2.98, 3.00, 2.98, 3.00, 3.04, 2.97, 3.02, 2.93, 3.03, 3.00, 2.98, 3.00, 3.02, 3.02, 2.98, 3.00, 3.03, 2.97, 3.02, 2.96, 3.02, 2.96, 3.02, 2.98, 2.98, 3.00, 2.99, 2.99, 3.02, 3.00, 3.00, 2.99, 2.99, 3.02, 2.96, 3.00, 2.98, 2.99, 3.04, 2.93, 3.02, 1.43, 1.07, 3.06, 3.00, 3.02, 2.98, 3.02, 2.95, 3.01, 3.02, 2.95, 2.99, 2.99, 3.01, 2.98, 2.99, 3.03, 2.99, 3.02, 2.96, 2.98, 3.01, 3.00, 3.00, 2.96, 2.99, 3.00, 3.00, 3.02, 2.96, 1.44];

y2hat = conv(flow2, hhat);
t2 = (1: length(flow2));

figure
hold on
plot(t2, y2hat(t2));
plot(t2, pressure2);
xlabel("time (sec)");
ylabel("pressure (kPa)");
title("Transfer Function Comparison on New Data");
legend('estimated', 'actual');
hold off

% Awful results. Abort mission!

%% Perform STATIC Estimation Using Poiseuille's Law
% Parameter Optimization for Tube Length and Viscosity Coeff's
% Adapted from main_LLM2.m given on Canvas.

Q = flow; % in mL/sec
P = pressure; % in kPa
n = 4.7; %viscosity (cP); best guess
l = 1; % tube length (m); best guess

nIn = (n/5:n/5:2*n);
lIn = (l/5:l/5:2*l);
nOut = zeros(10);
lOut = zeros(10);
jOut = zeros(10);


for i = 1:10 
    %Initial values of the model parameters
    params_init = [nIn(i), lIn(i)];

    %Optimize the model parameters
    options = optimset ('Display', 'off', 'PlotFcns', 'optimplotfval');
    [theta,Jmin] = fminsearch(@(params)(nmsePois(params,Q,P)), ...
    params_init, options);

    nOut(i) = theta(1);
    lOut(i) = theta(2);
    jOut(i) = Jmin;
end

%% STATIC Error Visualization 

r = 0.55; %tube radius for 20Gauge Catheter
t = (1:length(Q));

%Transfer function (poiseuille formula, in (kPa))
PHat = Q .* ((8*nOut(1)*lOut(1))/(pi*(r^4)));

%Predicted output ypred and cost J
e = PHat - P;
e2 = e .^ 2;

figure
hold on
plot(t,PHat);
plot(t,P);
xlabel("time (sec)");
ylabel("pressure (kPa)");
title("Static Estimate Visualization");
legend('Estimated', 'Actual');
hold off

figure
plot(t, e2);
title("Static Model Error");
xlabel("time (sec)");
ylabel("error (kPa)");

%While linear estimate was in the right ballpark, it did not account for
%the more gradual rise in pressure in response to a much quicker rise in
%flow (suggest a low-pass filter needs to be added).

%% Perform DYNAMIC Parameter Estimation Using Poiseuille's Law + LPF
% LPF = Low-Pass Filter
% Adapted from main_LLM2.m given on Canvas.

Q = flow; % in mL/sec
P = pressure; % in kPa

kIn = (n/5:n/5:2*n);
tauIn = (l/5:l/5:2*l);
kOut = zeros(10);
tauOut = zeros(10);
newJOut = zeros(10);
t = (1:length(Q));


for i = 1:10 
    %Initial values of the model parameters
    params_init = [kIn(i), tauIn(i)];

    %Optimize the model parameters
    options = optimset ('Display', 'off', 'PlotFcns', 'optimplotfval');
    [theta2,Jmin2] = fminsearch(@(params)(nmsePois3(params,Q,P,t)), ...
    params_init, options);

    kOut(i) = theta2(1);
    tauOut(i) = theta2(2);
    newJOut(i) = Jmin2;
end

%% DYNAMIC Model Error Visualization

%Transfer function
num = kOut(10);
den = [tauOut(10),1];
H = tf(num,den);

%Predicted output ypred and cost J
PHat = lsim(H,Q,t);
e3 = (P - PHat');
%e4 = e3 .^ 2;

figure
hold on
plot(t,PHat');
plot(t,P);
title("Dynamic Model Estimate");
xlabel("time (sec)")
ylabel("pressure (kPa)");
legend('Estimated', 'Actual');
hold off

figure
plot(t, e3);
title("Dynamic Model Error");
xlabel("time (sec)");
ylabel("error (kPa)");

% Shows overestimation. For our purposes, a conservatively high estimate
% does not hurt.


%% Testing Dynamic Estimation on Another Patient's Data

Q = flow2;
P = pressure2;

%Isolate 1st Phase
new_time = (time < 81);    % Focus on contrast phase to increase accuracy
Q = Q(new_time);
P = P(new_time);
time = time(new_time);

%Transfer function
num = kOut(10);
den = [tauOut(10),1];
H = tf(num,den);

%Predicted output ypred and cost J
PHat = lsim(H,Q,t);
e3 = (P - PHat');

figure
hold on
plot(t,PHat');
plot(t,P);
title("Dynamic Model Estimate on New Data");
xlabel("time (sec)")
ylabel("pressure (kPa)");
legend('Estimated', 'Actual');
hold off

figure
plot(t, e3);
title("Dynamic Model Error on New Data");
xlabel("time (sec)");
ylabel("error (kPa)");