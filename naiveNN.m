%% Import data
cd /Users/dwai/Documents/Box/MATLAB/Graduate/CS289A/ML_census
[Data Label] = ImportFile('adult_data');

%% Partition to test and training data
test_data = [Data(1:6500,1) Data(1:6500,13)];
test_label = Label(1:6500);
training_data = [Data(6501:end,1) Data(6501:end,13)];
training_label = Label(6501:end);

%% Plot Training Data

% figure(2), hold on
% for i = 1:length(training_label)
%   prediction = training_label(i);
%   x = training_data(i,1);
%   y = training_data(i,2);
%   if prediction == 1
%     h_1 = plot(x,y,'go');
%   else
%     h_0 = plot(x,y,'mo');
%   end
% end
% title('Training Data and their Labels')
% xlabel('Age'), ylabel('Hours per Week'), axis([10 90 0 100])

%% Train a kNN Model

mdl = ClassificationKNN.fit(training_data,training_label,'NumNeighbors',3,'Distance','cityblock');
predictions = predict(mdl,test_data);

%% Create Classified Regions and then Plot Results
xrange = 10:90;
yrange = 0:100;
[x,y] = meshgrid(xrange,yrange);
xy = [x(:) y(:)];

meshLabels = predict(mdl,xy);
decisionBoundaries = reshape(meshLabels,size(x));
figure(1), clf, hold on
h_img = imagesc(xrange,yrange,decisionBoundaries);
cmap = [166 136 163; 0 73 27]./255;
colormap(cmap);

for i = 1:length(predictions)
  prediction = test_label(i);
  x = test_data(i,1);
  y = test_data(i,2);
  if prediction == 1
    h_1 = plot(x,y,'go');
  else
    h_0 = plot(x,y,'mo');
  end
end
title('kNN Classifier, k =3, Manhattan distance metric') 
xlabel('Age'), ylabel('Hours per Week'), axis([10 90 0 100])
legend([h_1;h_0],'Income >50k','Income <=50k')

%% Determine Error Rate
incorrect = 0;
for i = 1:length(predictions)
  if predictions(i) ~= test_label(i)
    incorrect = incorrect + 1;
  end
end
error = incorrect/length(predictions)