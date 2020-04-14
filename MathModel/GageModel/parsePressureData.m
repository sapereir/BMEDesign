function [finalArray] = parsePressureData(RawData,gauge,flowRate,contrastType,SalineType,numDataPoints)
    %Import the t,P and Gauge data
    %gauge is a number (gauge of the catheter)
    %numDataPoints i the size of the dataset we want to work with
    %RawData is the path to the file we are parsing
    
    %This assumes that data is given in the format Bayer specified to us
    DataTable = RawData(:,12);
    GaugeTable = RawData(:,4);
    FlowRateContrastTable = RawData(:,6);
    
    DataCell = table2array(DataTable);
    GaugeCell = table2array(GaugeTable);
    FlowRateCell = table2array(FlowRateContrastTable);
    
    Data = DataCell(:,1);
    Gauge = GaugeCell(:,1);
    FlowRate = FlowRateCell(:,1);

    lenData = length(Data);
    i = 1;
    count = 0;
    tmpArray = [0,[],[]]*numDataPoints;

    while i < lenData && count < numDataPoints
        if Gauge(i,1) == gauge && parseFlowRate(FlowRate(i,1)) == flowRate
            %j is the true index in the overall data array
            count = count + 1;
            %Split time,pressure(t),flow(t) by ';'
            elem = Data(i);
            split = strsplit(elem,';');
            tRaw = split(1);
            pRaw = split(2);
            fRaw = split(3);

            %Take out the leading and trailing '[' and ']'
            tn = regexprep(tRaw,'[[]]','');
            pn = regexprep(pRaw,'[[]]','');
            fn = regexprep(fRaw,'[[]]','');

            %Now, turn tn,pn,fn into numerical arrays
            t = str2num(tn);
            p = str2num(pn);
            f = str2num(fn);
            %Length of t,p,f at this index
            len = length(t);
            
            %Now add i,t,p to the final Array
            for j = 1:len
                tmpArray(count,1,j) = i;
                tmpArray(count,2,j) = t(j);
                tmpArray(count,3,j) = p(j);
            end
        end
        i = i + 1;
    end
    
    finalArray = [0,[],[]]*numDataPoints;
    %Convert this into a cell array to make it easier to work with
    %Jagged 
    %Cells in cell array are vectors of certian lengths
        
    %Parse the data to truncate it and remove leading/trailing zeros
    for i = 1:numDataPoints
        deriv_pressure = diff(tmpArray(i,3,:));
        %deriv_pressure = [0 deriv_pressure];
        len = length(deriv_pressure);
        deriv_outliers_pressure = isoutlier(deriv_pressure);
        
        p_i = 1;
        while (p_i <= len) && ~((deriv_outliers_pressure(p_i) == 1) && ...
            (deriv_pressure(p_i) < 0))
            p_i = p_i + 1;
        end

        new_time = tmpArray(i,2,1:p_i-1);
        new_pressure = tmpArray(i,3,1:p_i-1);
        
        %Now remove the leading zeros
        start_index = 1;
        for j = 1:length(new_pressure)
            if new_pressure(j) == 0
                start_index = start_index + 1;
            end
        end
        
        %remove the trailing zeros
        end_index = length(new_pressure);
        for k = 1:length(new_pressure)-1
            if new_pressure(length(new_pressure)-k) == 0
                end_index = end_index - 1;
            end
        end
        
        length(new_pressure);
        start_index;
        end_index;
        newer_time = new_time(start_index:end_index);
        newer_pressure = new_pressure(start_index:end_index);
        
            %Add the results to the final array
        for j = 1:length(newer_time)
            finalArray(i,2,j) = newer_time(j);
            finalArray(i,3,j) = newer_pressure(j);
        end
        
    end
    %remove all the zeros from finalArray
    %finalArray is a big matrix with unknown number of zeroes
%     [sz1,sz2,sz3]=size(finalArray);         
%     for i=1:sz1
%         newest_time = nonzeros(finalArray(i,2,:));
%         newest_pressure = nonzeros(finalArray(i,3,:));
%         new_sz = length(newest_time);
%         finalArray(i,2) = zeros(1,length(newest_time));
%         finalArray(i,3) = zeros(1,length(newest_time));
%         
%         finalArray(i,2) = newest_time;
%         finalArray(i,3) = newest_pressure;
%     end
end