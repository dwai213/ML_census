%% Logistic Regression Classifier
% Using l-2 regulator logistic regression classifier
% What can be done: tuning params, different regulator, features(maybe?)
%**************************************************************************
clear all
clc
load 'Test_data.mat'
load 'Train_data.mat'
%**************************************************************************

% Data preprocessing(zero mean and identity variance)
Traindata = Train(:,[5 6]);
Trainlabel = TrainLabel;

Traindata = zscore(Traindata);
Traindata = [ones(size(Traindata,1),1) Traindata];

traindata = Traindata(1:10000,:);
validation = Traindata(10001:end,:);
trainlabel = Trainlabel(1:10000);
validationlabel = Trainlabel(10001:end);
% Updating parameter with stochastic gradient descent
iter = 40000;                               % Iteration times
C = 0.1;                                    % Learning rate constant
Lambda = [0.01 0.1 1];                               % Regularization constant
epsilon = 1e-6;
Accurate = [];
for l = 1:length(Lambda)
    lambda = Lambda(l);
    beta = ones(size(traindata,2),1);           % Initialize Beta
    beta_rec = beta;
    t = 1;                                      % Iteration counting
    loss{1} = 1;                                % Initialize loss function
    loss{2} = 0;
    loss_rec = [];
    
    while t<iter && abs(loss{2}-loss{1})>epsilon
        
        eta = C/sqrt(t);                                              % Learning rate
        idx = randi(size(traindata,1));                               % Randon index of data
        xtrain = traindata(idx,:);                                    % Random data
        mu = 1/(1+exp(-xtrain*beta));                                 % P(Y=1|x)
        G = -(trainlabel(idx)-mu) * xtrain';% + lambda * sign(beta) ;       % Gradient of loss function
%         H = 2 * lambda * eye(size(traindata,2)) + mu * (1-mu) * (xtrain * xtrain'); % Hessian of loss function
        beta = beta - eta * G;                                        % Beta update
        beta_rec = [beta_rec,beta];
        
        loss{1} = loss{2};
        tmp = 1-mu;                                                   % threshold for tiny 1-mu
        tmp(tmp<1e-10) = 1e-10;
        loss{2} =  -trainlabel(idx) * log(mu) - (1-trainlabel(idx))' * log(tmp);% + lambda * norm(beta,1); % Loss function
        loss_rec = [loss_rec,loss{2}];
        t = t+1;
    end
    P = 1 ./ (1 + exp(-validation * beta));
    P(P>=0.5) = 1;
    P(P<0.5) = 0;
    predLabel = P;
    Accurate(l) = sum(predLabel == validationlabel)/ length(validationlabel);
end
[~,I] = max(Accurate);
lambda = Lambda(I);
%% Prediction on train data
beta = ones(size(Traindata,2),1);
beta_rec = beta;
t = 1;
loss{1} = 1;
loss{2} = 0;
loss_rec = [];

while t<iter && abs(loss{2}-loss{1})>epsilon
    
    eta = C/sqrt(t);                                              % Learning rate
    idx = randi(size(Traindata,1));                               % Randon index of data
    xtrain = Traindata(idx,:);                                    % Random data
    mu = 1/(1+exp(-xtrain*beta));                                 % P(Y=1|x)
    G = -(Trainlabel(idx)-mu) * xtrain';% + lambda * sign(beta) ;       % Gradient of loss function
%     H = 2 * lambda * eye(size(Traindata,2)) + mu * (1-mu) * (xtrain * xtrain'); % Hessian of loss function
    beta = beta - eta * G;                                        % Beta update
    beta_rec = [beta_rec,beta];
    
    loss{1} = loss{2};
    tmp = 1-mu;                                                   % threshold for tiny 1-mu
    tmp(tmp<1e-10) = 1e-10;
    loss{2} = -Trainlabel(idx) * log(mu) - (1-Trainlabel(idx))' * log(tmp);% + lambda * norm(beta,1) ; % Loss function
    loss_rec = [loss_rec,loss{2}];
    t = t+1;
end

P = 1 ./ (1 + exp(-Traindata * beta));
P(P>=0.5) = 1;
P(P<0.5) = 0;

predLabel = P;

% Error rate

Accurate = sum(predLabel == Trainlabel)/ length(Trainlabel);

[~,I] = sort(abs(beta),'descend');
% I = I(1:3);
save('5_6.mat','predLabel','Accurate','I')
%% Prediction on test data
Testdata = Test(:,[5 6]);
Testdata = zscore(Testdata);
Testdata = [ones(size(Testdata,1),1), Testdata];

testlabel = 1 ./ (1 + exp(-Testdata * beta));

testlabel(testlabel>=0.5) = 1;
testlabel(testlabel<0.5) = 0;

Accurate_test = sum(testlabel == TestLabel)/ length(TestLabel);
save('5_6_Test.mat','testlabel','Accurate_test')
