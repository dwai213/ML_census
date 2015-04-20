%% CS289 Project
% Data Transformation
TrainFile = 'adult_data';
TestFile = 'adult_test';
[Train,TrainLabel] = ImportFile(TrainFile);
[Test, TestLabel] = ImportFile(TestFile);
save('ML_census/Train_data.mat','Train','TrainLabel');
save('ML_census/Test_data.mat','Test','TestLabel');
