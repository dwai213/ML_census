function error = findError(predictions,correct_labels)
%Helper function to return the prediction error given the correct
%labels

error = 0;
for i = 1:length(predictions)
  if predictions(i) ~= correct_labels(i)
    error = error + 1;
  end
end
error = error/length(predictions)*100;

end