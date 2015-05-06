%% Perceptron.m
%% Maintainer: Dennis Wai
%  This code is the master script to run a simple perceptron
%  classifier on the adult data

%% Load Data
clear all, clc, close all
cd /Users/dwai/Documents/ML_census
load Train_data.mat

%% Partition Data into Sets
r_ind = randperm(length(Train));
valid_size = 5000;
valid_data = Train(r_ind(1:valid_size),:);
valid_label = TrainLabel(r_ind(1:valid_size),:);
train_data = Train(r_ind((valid_size+1):end),:);
train_label = TrainLabel(r_ind((valid_size+1):end),:);

%make train_label +/- 1
train_label(train_label==0)=-1;
valid_label(valid_label==0)=-1;

%% Perceptron Training

hasConverged = 0;
subset = [1:4 6:14];
[N,d] = size(train_data(:,subset));
percept_data = [train_data(:,subset) ones(N,1)];
i = 0;

H = 10; %error array history
err_hist = zeros(1,H);
th = random('norm',0,.1,d+1,1); %initialized weight
tol = .1;

% percept_data = [2 2 1; 4 2 1; 2 4 1; 4 4 1];
% train_label = [1;-1;1;-1];
[N,~] = size(percept_data);

i = 0;
while ~hasConverged
  ind = mod(i,N)+1;
  e_ind = mod(i,H)+1;
  
  p = percept_data*th;
  p(p >= 0) = 1; p(p < 0) = -1;
  err_hist(e_ind) = errorRate(p,train_label);
  if (abs(mean(diff(err_hist))) <= tol) && i > N
    hasConverged = 1;
  end
  
  sample = percept_data(ind,:);
  label = train_label(ind);
  product = th'*sample';
  if (product >= 0 && label>=0) || (product < 0 && label<0)
    %correctly classified
    th = th;
  else
    %incorrectly classified
    th = th + label*sample';
  end
  i = i + 1;
  
  if mod(i,10) == 0
    disp(sprintf('Iter: %d Training Acc: %2.5f',i,1-err_hist(e_ind)))
  end
end
disp('Done Training')

%% Validation

p = [valid_data(:,subset) ones(valid_size,1)]*th;
p(p >= 0) = 1; p(p < 0) = -1;
disp(sprintf('Validating Acc: %3.3f%%',(1-errorRate(p,valid_label))*100))

%% Plotting Data
% if length(subset) == 3
%   figure(1), clf, hold on
%   for i = 1:N
%     x = percept_data(i,1);
%     y = percept_data(i,2);
%     z = percept_data(i,3);
%     if train_data(i) == 1
%       plot3(x,y,z,'o');
%     else
%       plot3(x,y,z,'x');
%     end
%   end
%   grid on
%   xlabel('X'); ylabel('Y'); zlabel('Z');
%   view(45,30)
%   box on
%   axis tight
% end

if length(subset) == 2
  figure(1), clf, hold on
  plot_data = [percept_data train_label];
  x_pos = plot_data((plot_data(:,end) == 1),1);
  y_pos = plot_data((plot_data(:,end) == 1),2);
  x_neg = plot_data((plot_data(:,end) == -1),1);
  y_neg = plot_data((plot_data(:,end) == -1),2);
  plot(x_pos,y_pos,'go');
  plot(x_neg,y_neg,'rx');
  grid on
  xlabel('X'); ylabel('Y'); title('Training Accuracy');
  
  x = min(plot_data(:,1)):max(plot_data(:,1));
  y = (-th(1)*x-th(3))/th(2);
  plot(x,y,'k--')
  axis([x(1) x(end) min(plot_data(:,2)) max(plot_data(:,2))])
end

if length(subset) == 2
  figure(2), clf, hold on
  plot_data = [valid_data(:,subset) valid_label];
  x_pos = plot_data((plot_data(:,end) == 1),1);
  y_pos = plot_data((plot_data(:,end) == 1),2);
  x_neg = plot_data((plot_data(:,end) == -1),1);
  y_neg = plot_data((plot_data(:,end) == -1),2);
  plot(x_pos,y_pos,'go');
  plot(x_neg,y_neg,'rx');
  grid on
  xlabel('X'); ylabel('Y'); title('Validating Accuracy')
  
  x = min(plot_data(:,1)):max(plot_data(:,1));
  y = (-th(1)*x-th(3))/th(2);
  plot(x,y,'k--')
  axis([x(1) x(end) min(plot_data(:,2)) max(plot_data(:,2))])
end
