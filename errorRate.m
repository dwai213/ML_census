function error = errorRate(predictions,labels)
% errorRate
% Given a vector or predictions and the correct labels, returns
% the error rate

error = 1-sum((predictions == labels))/length(labels);

end