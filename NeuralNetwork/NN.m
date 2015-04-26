%% Neural NetWork
%**************************************************************************
clear all
clc
load 'Test_data.mat'
load 'Train_data.mat'
%**************************************************************************

% Data preprocessing(zero mean and identity variance)
traindata = Train;
trainlabel = TrainLabel;

traindata = zscore(traindata);

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
for learnRate = 1: length(eta)
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
        validLabel = labels(validIdx,:);
        trainLabel = labels(trainIdx,:);
        
        % Training and error
        [W_train,w_train] = train_mulNN(trainSet,trainLabel,{eta(learnRate), errorFcn(1),maxiter});
        predict = predictNN(W_train,validSet);
        Error(set) = sum(predict~=validLabel-1)/validNum;
    
    end
    
    % Average error
    Error_eta(learnRate) = mean(Error);
end
validationTime = toc;

[~,eta_idx] = min(Error_eta);
BestLearningRate = eta(eta_idx);
% 
% % Save parameters
% save('valid_ms_2.mat','Error_eta','BestLearningRate','validationTime');

%% Training on whole training set
maxiter = 100000;
errorFcn = {'mean_squared','cross_entropy'};
tic;
%*************** Adjust number of layers anf number of neurons************ 
n_hid = [200 200 100];
%*************************************************************************
[W,w] = train_mulNN(traindata,labels,{BestLearningRate,errorFcn(2),maxiter,n_hid});
predict = NNpredict(W,traindata);
error = sum(predict~=trainlabel)/size(traindata,1);
trainingTime = toc;

%% Testing data
test_label = predictNN(W,test);