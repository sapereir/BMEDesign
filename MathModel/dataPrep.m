function [q_err, new_time, new_pressure,new_flow] = dataPrep(max_q, time,flow,...
    pressure)
%DATAPREP Function Version of data_preprocess.m

deriv_pressure = diff(pressure);
deriv_pressure = [0 deriv_pressure];
deriv_flow = diff(flow);
deriv_flow = [0 deriv_flow];

%% Truncating at end of first steady state

% Loop through the vector to detect significant change in derivative
% (outliers). Specifically, first significant change in derivative that is
% negative should mark the end of the first steady state.

%Visualize outliers in derivative
deriv_outliers_flow = isoutlier(deriv_flow);
deriv_outliers_pressure = isoutlier(deriv_pressure);

qstop = size(flow);
qstop = qstop(2); %get length of the data vector
q_i = 1;
while (q_i <= qstop) && ~((deriv_outliers_flow(q_i) == 1) && ...
    (deriv_flow(q_i) < 0))
    q_i = q_i + 1;
end

new_flow = flow(1:q_i-1);
new_time = time(1:q_i-1);
new_pressure = pressure(1:q_i-1);
  
%% Truncating at Beginning of First Steady-State

% Using derivative as before, but looping backward from end of the
% first-phase to when there's a significant change in derviative

q_i = q_i - 1;
while (q_i > 0) && ~((deriv_outliers_flow(q_i) == 1)) 
    q_i = q_i - 1; %loop bacwards
end

new_flow = new_flow(q_i:end);
new_time = new_time(q_i:end);
new_pressure = new_pressure(q_i:end);
%% Using Unit Step as a Reference

%unit step scaled by max flow for reference 
ref_q = zeros(1,length(new_flow));
ref_q = ref_q + max_q;
q_err = abs((ref_q-new_flow)./ref_q);
end

