%% Flow Prediction
% Using Step Function
% Takes in RawData imported into MATLAB:
%      - Change name of procedure column to 'Procedure'
%      - Change var type of procedure column to string
%      - delimited by tab and semicolon
%      - imported directly as table

%% Isolating Data Vectors and Values

row = 15; %row number
max_q = RawData(row, 6);
time = RawData(row, 12);
pressure = RawData(row, 13);
flow = RawData(row, 14);

%Convert from table to string to int or int vector
flow = flow.Flow;
flow = textscan(flow, '%f', 'Delimiter',',' );
flow = flow{1, 1}';

pressure = pressure.Pressure;
pressure = textscan(pressure, '%f', 'Delimiter',',');
pressure = pressure{1, 1}';

time = time.Time;
time = strip(time,'left','[');
time = textscan(time, '%f', 'Delimiter', ',');
time = time{1, 1}';

max_q = max_q.Procedure;
max_q = strip(max_q,'left','@');
max_q = textscan(max_q, '%f', 'Delimiter', ' ');
max_q = max_q(1);
max_q = max_q{1,1}'; 
