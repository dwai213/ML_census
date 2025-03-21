%% Import data from text file.
% function [Data,Label] = ImportFile(FileName)

% Script for importing data from the following text file:
%
%    /Users/sony/Dropbox/CS289Project/adult/adult_data.txt
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2015/04/03 16:00:12

%% Initialize variables.
filename = [pwd '/adult/' FileName '.txt'];
delimiter = ',';

%% Format string for each line of text:
%   column1: double (%f)
%	column2: text (%s)
%   column3: double (%f)
%	column4: text (%s)
%   column5: double (%f)
%	column6: text (%s)
%   column7: text (%s)
%	column8: text (%s)
%   column9: text (%s)
%	column10: text (%s)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: text (%s)
%   column15: text (%s)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%s%f%s%f%s%s%s%s%s%f%f%f%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN, 'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% Convert discrete features into discrete integers via hash
% tables
classes = cell(8,1);
%workclass
classes{1} = {'?','Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'};
%education
classes{2} = {'Doctorate','Masters','Bachelors', 'Some-college','HS-grad','Prof-school','Assoc-acdm','Assoc-voc','12th','11th','10th','9th','7th-8th','5th-6th','1st-4th','Preschool','?'};
%marital-status
classes{3} = {'?','Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'};
%occupation
classes{4} = {'?','Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'};
%relationship
classes{5} = {'?','Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'};
%race
classes{6} = {'?','White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'};
%sex
classes{7} = {'?','Female', 'Male'};
%native-country
classes{8} = {'?','United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran','Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'};
tables = cell(8,1);
for i = 1:8
  tables{i} = makeHashTables(classes{i});
end


%% Allocate imported array to column variable names
data.AGE = dataArray{:, 1};
data.WORKCLASS = convertClass2Integer(dataArray{:, 2},tables{1});
data.FNLWGT = dataArray{:, 3};
data.EDUCATION = convertClass2Integer(dataArray{:, 4},tables{2});
data.EDUCATION_NUM = dataArray{:, 5};
data.MARITAL = convertClass2Integer(dataArray{:, 6},tables{3});
data.OCCUPATION = convertClass2Integer(dataArray{:, 7},tables{4});
data.RELATIONSHIP = convertClass2Integer(dataArray{:, 8},tables{5});
data.RACE = convertClass2Integer(dataArray{:, 9},tables{6});
data.SEX = convertClass2Integer(dataArray{:, 10},tables{7});
data.CAPITAL_GAIN = dataArray{:, 11};
data.CAPITAL_LOSS = dataArray{:, 12};
data.HOURS_PER_WEEK = dataArray{:, 13};
data.NATIVE_COUNTRY = convertClass2Integer(dataArray{:, 14},tables{8});
data.LABEL = dataArray{:, 15};

Label = data.LABEL;
data = rmfield(data,'LABEL');

for i = 1:length(Label) 
   if strcmp(Label{i},'<=50K') || strcmp(Label{i},'<=50K.')
       Label{i} = 0;  
   else
       Label{i} = 1;
   end
end
Label = cell2mat(Label);

field = fields(data);
Data = [];
for j = 1:length(field)
    Data = [Data,data.(field{j})];
end    
%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans;