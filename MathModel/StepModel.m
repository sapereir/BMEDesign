clc;clear;
%import data from John
data = readtable('SN-SRMC-CT-1.txt');
%remove columns indicating protocol common amount all data points
datanew = removevars(data,{'Var1', 'Var2', 'Var3','Var11'});
%sort data by 'Abort'
datanewabort = sortrows(datanew,6);
datanew0 = removevars(datanewabort,{'Var9'});
datanew1 = datanew0(94:59346,:);
datanewnew = sortrows(datanew1,6);
datatemp = datanewnew(201:59253,:);
data2 = removevars(datatemp,{'Var10','Var7','Var8'});
%sort data by Gauge Type
datanew2 = sortrows(data2,1);
%Central Line
centralline = datanew2(1:38,:);
%Gauge16
gauge16 = datanew2(39:44,:);
%Gauge 18
gauge18 = datanew2(45:1894,:);
%Gauge 20
gauge20 = datanew2(1895:38430,:);
%Gauge 22
gauge22 = datanew2(38431:53673,:);
%Gauge 23
gauge23 = datanew2(53674:53675,:);
%Gauge 24
gauge24 = datanew2(53676:54662,:);

%% Isolating Data Vectors and Values For Gauge 22

row_num = 25; %row number
row = gauge22(row_num,:);
max_q = row.Var6;

data = row.Var12{1,1};
data = convertCharsToStrings(data);
data = strip(data,'left','[');
data = strip(data,'right','[');
data = split(data, ';');
time = data(1);
pressure = data(2);
flow = data(3);

% Convert from string to int or int vector

flow = textscan(flow, '%f', 'Delimiter',',' );
flow = flow{1, 1}';

pressure = textscan(pressure, '%f', 'Delimiter',',');
pressure = pressure{1, 1}';

time = textscan(time, '%f', 'Delimiter', ',');
time = time{1, 1}';

max_q = max_q{1,1};
max_q = strip(max_q,'left','@');
max_q = textscan(max_q, '%f', 'Delimiter', ' ');
max_q = max_q(1);
max_q = max_q{1,1}'; 

%% Preprocess Data

[q_err, newest_time, newest_pressure,newest_flow] = ...
    dataPrep(max_q, time, flow, pressure);

figure
plot(newest_time, q_err);
title("Magnitude of Flow Error Using Procedure as Reference");
xlabel("time (s)");
ylabel("Error");

