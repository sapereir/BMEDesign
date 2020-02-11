%% Data Preprocessing 

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
