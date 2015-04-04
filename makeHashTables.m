function hashTable = makeHashTables(classes)
%This function makes a hash table that goes from a String Field
%to an integer
keys = classes;
values = 0:(length(classes)-1);

hashTable = containers.Map(keys, values);

end