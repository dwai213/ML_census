%% Dennis Wai 2118 3965
%% Load data
clear all, clc, close all
cd /Users/dwai/Documents/ML_census/
load Train_data.mat
cd /Users/dwai/Documents/ML_census/trees/

%loaded a priori
features = {'age';'workclass';'fnlwgt';'education';...
  'education-num';'marital-status';'occupation';'relationship';...
  'race';'sex';'capital-gain';'capital-loss';'hours-per-week';...
  'native-country'};

%% Partition Data
r_ind = randperm(length(Train));
valid_size = 5000;
valid_data = Train(r_ind(1:valid_size),:);
valid_labels = TrainLabel(r_ind(1:valid_size),:);
train_data = Train(r_ind((valid_size+1):end),:);
train_labels = TrainLabel(r_ind((valid_size+1):end),:);

%% Train a Decision Tree
subset = [1:2 5:14];
feature_vector = {};
for i = subset
  feature_vector = [feature_vector; features{i}];
end
tree_data = train_data(:,subset);
clc
tic
trainedTree = DecisionTree(feature_vector,4);
trainedTree.train(tree_data,train_labels');
root = trainedTree.root;
toc

%% Predictions with Tree
% [n d] = size(test_data);
% predictions = trainedTree.predict(test_data);
% error = findError(predictions,ones(n,1))
[n d] = size(valid_data(:,subset));
predictions = trainedTree.predict(valid_data(:,subset));
disp(sprintf('Validation Acc: %2.5f',100-findError(predictions,valid_labels)))

%% Example Prediction with Tree
decisions = trainedTree.predictVerbose(valid_data(1200,subset));
disp(decisions(:))

%% Export Tree for graphviz to Visualize
trainedTree.export()
display 'Tree exported'