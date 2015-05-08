% %% Dennis Wai 2118 3965
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

subset = [1:2 5:14];
feature_vector = {};
for i = subset
  feature_vector = [feature_vector; features{i}];
end

%% Train a Decision Tree
tree_data = train_data(:,subset);
clc
tic
%num trees, # features for splits, depth limit
trainedForest = RandomForest(10,4,10,feature_vector);
forest = trainedForest.train(tree_data,train_labels');
trees = trainedForest.Trees
toc

%% Most Common Top Node Splits
a = zeros(trainedForest.NumTrees,1);
for i = 1:trainedForest.NumTrees
  a(i) = trainedForest.Trees{i}.root.split_rule(1);
end
[uniqs,ind,~] = unique(sort(a));
temp = diff(ind);
for i = 1:length(uniqs)
  if i == 1
    freq = ind(i);
  else
    freq = temp(i-1);
  end
  fprintf('%s showed up %d times\n',feature_vector{uniqs(i)},freq)
end

%% Predictions with Tree
% [n d] = size(test_data);
% predictions = trainedForest.predict(test_data);
% error = findError(predictions,ones(n,1))
[n d] = size(valid_data(:,subset));
predictions = trainedForest.predict(valid_data(:,subset));
disp(sprintf('Validation Acc: %2.5f',100-findError(predictions,valid_labels)))




%% Export Tree for graphviz to Visualize
trees{1}.export()
display 'tree exported'