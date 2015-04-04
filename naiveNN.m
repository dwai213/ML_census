%% Import data
[Data Label] = ImportFile('adult_data');

%% Partition to test and training data
test_data = [Data.AGE(1:6500) Data.HOURS_PER_WEEK(1:6500)];
test_label = Label(1:6500);
training_data = [Data.AGE(6501:end) Data.HOURS_PER_WEEK(6501:end)];
training_label = Label(6501:end);

%% Plot Training Data

figure(2), hold on
for i = 1:length(training_label)
  prediction = training_label(i);
  x = training_data(i,1);
  y = training_data(i,2);
  if prediction == 1
    plot(x,y,'go')
  else
    plot(x,y,'mo')
  end
end
title('Training Data and their Labels')
xlabel('Age'), ylabel('Hours per Week'), axis([10 90 0 100])

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
imagesc(xrange,yrange,decisionBoundaries)
cmap = [166 136 163; 0 73 27]./255;
colormap(cmap);

for i = 1:length(predictions)
  prediction = test_label(i);
  x = test_data(i,1);
  y = test_data(i,2);
  if prediction == 1
    plot(x,y,'go')
  else
    plot(x,y,'mo')
  end
end
title('Test Data Superimposed on Decision Boundary') 
xlabel('Age'), ylabel('Hours per Week'), axis([10 90 0 100])

%% Determine Error Rate
incorrect = 0;
for i = 1:length(predictions)
  if predictions(i) ~= test_label(i)
    incorrect = incorrect + 1;
  end
end
error = incorrect/length(predictions)