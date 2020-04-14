clc;clear;
%import data from John
data = readtable('SN-SRMC-CT-1 3.txt');
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

