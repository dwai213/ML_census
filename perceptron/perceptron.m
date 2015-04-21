%% Perceptron.m
%% Maintainer: Dennis Wai
%  This code is the master script to run a simple perceptron
%  classifier on the adult data

%% Load Data
clear all, clc
cd /Users/dwai/Documents/Box/MATLAB/Graduate/CS289A/ML_census
load Train_data.mat

%% Partition Data into Sets
r_ind = randperm(length(Train));
valid_size = 3200;
valid_data = normr(Train(r_ind(1:valid_size),:));
valid_label = TrainLabel(r_ind(1:valid_size),:);
train_data = normr(Train(r_ind((valid_size+1):end),:));
train_label = TrainLabel(r_ind((valid_size+1):end),:);

%make train_label +/- 1
train_label(train_label==0)=-1;

%% Perceptron Training

misclassification = 1;
[N d] = size(train_data);
i = 0;

H = 10; %error array history
err_hist = zeros(1,H);
th = ones(d,1); %initialized weight
tol = .1;

while misclassification
  ind = mod(i,N)+1;
  e_ind = mod(i,H)+1;
  
  p = train_data*th;
  p = (p > 0); p(p==0) = -1;
  err_hist(e_ind) = errorRate(p,train_label);
  if i > N
    misclassification = (abs(mean(diff(err_hist))) > tol);
  end
  
  sample = train_data(ind,:);
  label = train_label(ind);
  product = th'*sample';
  if (product > 0 && label>0) || (product < 0 && label<0)
    %correctly classified
    th = th;
  else
    %incorrectly classified
    th = th + label*sample';
  end
  i = i + 1;
  if mod(i,3000) == 0
    sprintf('Iter: %d Err: %2.5f',i,err_hist(e_ind))
    figure(1)
    stem(th)
    axis([0 15 -5 5])
  end
end
disp('Done Training')

%% Validation

p = valid_data*th;
p = (p > 0); p(p==0) = -1;
sprintf('Err: %3.3f%%',errorRate(p,valid_label)*100)

