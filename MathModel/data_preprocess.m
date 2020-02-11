%% Data Preprocessing 
% By Shayan "Ricky" Gupta (shayang)

%% Script Description
% In this script, the raw flow and pressure data will be prepocessed by (1)
% removing zeros in the beginning of the data (aka time delay) and (2)
% truncating the data at the end of the steady-state of the fist (assumed
% more viscous, therefore higher pressure) phase.

%% Load and Visualize One Raw Data Set

%% Load Data

time = [0.000, 0.355, 0.591, 0.877, 1.230, 1.528, 1.878, 2.174, 2.569, 2.880, 3.133, 3.431, 3.731, 4.031, 4.327, 4.599, 4.853, 5.103, 5.356, 5.637, 5.968, 6.279, 6.628, 6.985, 7.268, 7.528, 7.872, 8.195, 8.458, 8.780, 9.167, 9.518, 9.829, 10.090, 10.347, 10.590, 10.840, 11.118, 11.366, 11.640, 11.863, 12.114, 12.352, 12.608, 12.832, 13.083, 13.356, 13.601, 13.857, 14.106, 14.448, 14.690, 14.985, 15.317, 15.610, 15.869, 16.124, 16.454, 16.767, 17.077, 17.324, 17.612, 17.875, 18.174, 18.523, 18.861, 19.124, 19.476, 19.714, 19.984, 20.244, 20.482, 20.736, 21.067, 21.367, 21.676, 21.973, 22.329];
pressure = [0, 0, 69, 145, 234, 296, 496, 600, 814, 903, 993, 1076, 1165, 1207, 1276, 1282, 1317, 1317, 1372, 1379, 1393, 1400, 1413, 1434, 1400, 1427, 1448, 1420, 1427, 1400, 1434, 1434, 1434, 1420, 1413, 1420, 1407, 1420, 1393, 1441, 1372, 1393, 1365, 1420, 1372, 1365, 1358, 1393, 1365, 1365, 1331, 1358, 1331, 1358, 1317, 1331, 1358, 1262, 1069, 834, 910, 924, 889, 903, 834, 779, 752, 696, 676, 655, 621, 607, 621, 593, 586, 579, 517, 352];
flow = [0.00, 0.12, 4.13, 6.10, 5.99, 5.94, 6.02, 5.94, 5.99, 6.10, 5.86, 6.03, 6.00, 5.94, 6.01, 6.06, 5.93, 6.01, 6.04, 5.98, 5.97, 5.94, 6.03, 5.99, 5.99, 5.99, 5.99, 6.06, 5.88, 6.07, 6.00, 5.90, 6.00, 6.01, 5.92, 6.07, 5.93, 6.10, 5.93, 5.97, 6.03, 5.90, 5.96, 6.05, 5.98, 6.03, 5.97, 6.01, 5.97, 5.96, 6.04, 6.01, 5.95, 5.97, 5.98, 6.00, 5.99, 5.33, 2.26, 0.62, 4.84, 6.01, 5.98, 6.00, 5.98, 5.98, 6.06, 5.97, 6.01, 5.98, 6.02, 5.98, 5.97, 6.01, 5.92, 6.10, 5.08, 0.98];

% Visualize Raw Data
figure %pressure data
subplot(2,1,1); 
    plot(time, pressure);
    xline(16.12); %end of steady state by visual inspection
                  %(around where part of preprocessing should occur)
    xlabel('time(sec)');
    ylabel('pressure(kPa)');
    title('Catheter Flow and Pressure Data');
    
subplot(2,1,2); % flow data
    plot(time, flow);
    xline(16.12);
    xlabel('time(sec)');
    ylabel('flow rate(mL/s)');
    
% Visualize Raw Data's Derivative
% Ricky: I thought it might be useful to use the derivative because I
% expect the end of the first steady state to have a significant
% discontinuity in the derivative.
deriv_pressure = diff(pressure);
deriv_pressure = [0 deriv_pressure];
deriv_flow = diff(flow);
deriv_flow = [0 deriv_flow];

figure %pressure data
subplot(2,1,1); 
    plot(time,deriv_pressure);
    xline(16.12); %end of steady state by visual inspection
                  %(around where part of preprocessing should occur)
    xlabel('time(sec)');
    ylabel('d/dt(pressure(kPa))');
    title('Catheter Flow and Pressure Data');
    
subplot(2,1,2); % flow data
    plot(time,deriv_flow);
    xline(16.12);
    xlabel('time(sec)');
    ylabel('d/dt(flow rate(mL/s))');
    
%% Part 1: Removing Time Delay
% Loop through the beginning of the vector

qstop = size(flow);
qstop = qstop(2); %get length of the data vector

q_i = 1;
while (q_i <= qstop) && (flow(q_i) == 0)
    q_i = q_i + 1;
end

%repeat for pressure
pstop = size(pressure);
pstop = pstop(2); %get length of the data vector

p_i = 1;
while (p_i <= pstop) && (pressure(p_i) == 0)
    p_i = p_i + 1;
end

i = max(p_i, q_i);
new_flow = flow(i:end);
new_pressure = pressure(i:end);
new_time = time(i:end);

% Visualize Partially pre-processed Data
figure %pressure data
subplot(2,1,1); 
    plot(new_time, new_pressure);
    xline(16.12); %end of steady state by visual inspection
                  %(around where part of preprocessing should occur)
    xlabel('time(sec)');
    ylabel('pressure(kPa)');
    title('Catheter Flow and Pressure Data');
    
subplot(2,1,2); % flow data
    plot(new_time, new_flow);
    xline(16.12);
    xlabel('time(sec)');
    ylabel('flow rate(mL/s)');
