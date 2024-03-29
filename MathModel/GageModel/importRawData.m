function RawData = importRawData(filename, dataLines)
%IMPORTFILE Import data from a text file
%  RAWDATA = IMPORTFILE(FILENAME) reads data from text file FILENAME for
%  the default selection.  Returns the data as a table.
%
%  RAWDATA = IMPORTFILE(FILE, DATALINES) reads data for the specified
%  row interval(s) of text file FILENAME. Specify DATALINES as a
%  positive scalar integer or a N-by-2 array of positive scalar integers
%  for dis-contiguous row intervals.
%
%  Example:
%  RawData = importfile("C:\Users\illum\Documents\School\Spring 20\BME Design\BME Design Math Model Attempt 3\MathModel\GageModel\RawData.txt", [1, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 13-Feb-2020 18:35:20

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [1, Inf];
end

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 12);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["VarName1", "VarName2", "ISOVUE300", "Gauge20", "LeftAntecubital", "C97S30", "fe1939ec9048b993016f9c571c201b", "T1456431050000Z", "Normal", "None", "VarName11", "VarName12"];
opts.VariableTypes = ["string", "double", "double", "double", "categorical", "categorical", "string", "string", "categorical", "categorical", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["VarName1", "fe1939ec9048b993016f9c571c201b", "T1456431050000Z", "VarName12"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["VarName1", "LeftAntecubital", "C97S30", "fe1939ec9048b993016f9c571c201b", "T1456431050000Z", "Normal", "None", "VarName12"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, ["ISOVUE300", "Gauge20"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["ISOVUE300", "Gauge20"], "ThousandsSeparator", ",");

% Import the data
RawData = readtable(filename, opts);

end