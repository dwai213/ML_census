function convertedData = convertClass2Integer(raw_data,table)
%This function converts the String feature in raw_data and
%uses the provided table to convert it to an integer

convertedData = zeros(size(raw_data));
for i = 1:length(raw_data)
  convertedData(i) = table(raw_data{i});
end

end