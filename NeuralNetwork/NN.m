%% Neural NetWork
%**************************************************************************
clear all
clc
load 'Test_data.mat'
load 'Train_data.mat'
%**************************************************************************

% Data preprocessing(zero mean and identity variance)
traindata = Train(:,[5 6]);
trainlabel = TrainLabel;
test = Test(:,[5 6]);
traindata = zscore(traindata);

test = zscore(test);
% Augment Label 
tmp = [];
for i = 1:length(trainlabel)
   augment = zeros(1,2);
   if trainlabel(i)==1
      augment(2) = 1;                                      % Augmented label vector
      tmp = [tmp;augment];
   else
      augment(1) = 1;
      tmp = [tmp;augment];
   end
end

labels = tmp;      

%% Validation for tuning
% Error Function Selection
errorFcn = {'mean_squared','cross_entropy'};

% Tuning on training set
tic;
eta = [0.01, 0.1, 1];                           % Tuning learning rate
maxiter = 100000;                               % Max iteration time
idx = randperm(32560);
Error = [];                                     % Cross-validation error
Error_eta = [];                                 % Error of different learning rate
NumSet = 10;                                    % #-fold cross validation
n_hid = [200];
lambda = [0.01 0.1 1];
for learnRate = 1: length(eta)
    for penalterm = 1:length(lambda)
        for set = 1:NumSet
            validNum = 32560/NumSet;
            TrainData = traindata;
            
            % Index
            validIdx = idx((set-1)*validNum+1 : set*validNum);
            trainIdx = idx;
            trainIdx((set-1)*validNum+1 : set*validNum) = [];
            
            % Validation set and training set
            validSet = TrainData(validIdx,:);
            trainSet = TrainData(trainIdx,:);
            validLabel = trainlabel(validIdx,:);
            trainLabel = labels(trainIdx,:);
            
            % Training and error
            [W_train,w_train] = train_mulNN(trainSet,trainLabel,{eta(learnRate), errorFcn(2),maxiter,n_hid,lambda(penalterm)});
            predict = NNpredict(W_train,validSet);
            Error(set) = sum(predict~=validLabel)/validNum;
            
        end
        
        % Average error
        Error_eta(learnRate,penalterm) = mean(Error);
    end
end
validationTime = toc;

[~,IDX] = min(Error_eta(:));
[BestLearningRate,Penalty] = ind2sub(size(Error_eta),IDX);
BestLearningRate = eta(BestLearningRate);
Penalty = lambda(Penalty);

% Save parameters
save('5_6valNN2.mat','Error_eta','BestLearningRate','Penalty','validationTime','n_hid');

%% Training on whole training set
maxiter = 100000;
errorFcn = {'mean_squared','cross_entropy'};
tic;
%*************** Adjust number of layers anf number of neurons************ 
% n_hid = [200];
%*************************************************************************
[W,w] = train_mulNN(traindata,labels,{BestLearningRate,errorFcn(2),maxiter,n_hid,Penalty});
predict = NNpredict(W,traindata);
error = sum(predict~=trainlabel)/size(traindata,1);
trainingTime = toc;
save('5_6NNtrain2.mat','W','w','error','trainingTime','n_hid');

%% Testing data
test_label = NNpredict(W,test);
error = sum(test_label~=TestLabel)/size(test,1);
save('5_6NN_test2.mat','error','test_label')