test_data = Test(:,[1,13]);
test_label = TestLabel;
training_data = Train(1:6500,[1,13]);
training_label = TrainLabel(1:6500);

%% Plot Training Data

% figure(2), hold on
% for i = 1:length(training_label)
%   prediction = training_label(i);
%   x = training_data(i,1);
%   y = training_data(i,2);
%   if prediction == 1
%     plot(x,y,'go')
%   else
%     plot(x,y,'mo')
%   end
% end
% title('Training Data and their Labels')
% xlabel('Age'), ylabel('Hours per Week'), axis([10 90 0 100])
% figure(2)
%  X = 10:90;
%  w = [1.4684 -0.0117 -0.0127];
%  Y =( -w(2)*X-w(1))/w(3);
%  plot(X,Y)
 
 
figure(3), hold on
for i = 1:length(test_label)
  prediction = test_label(i);
  x = test_data(i,1);
  y = test_data(i,2);
  if prediction == 1
    p1 = plot(x,y,'go');
    
  else
    p2 = plot(x,y,'mo');
  end
   
end

title('SVM Classifier')
xlabel('Age'), ylabel('Hours per Week'), axis([10 90 0 100])
 X = 10:90;
 w = [1.4684 -0.0117 -0.0127];
 Y =( -w(2)*X-w(1))/w(3);
 p3 = plot(X,Y)
legend([p1,p2,p3],'income > 50k','income <= 50k','Decision Boundary')