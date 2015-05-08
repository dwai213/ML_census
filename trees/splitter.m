function splitData = splitter(ind,labels,data)
% Returns the subset of data/labels at ind. Also provides the
% amount of positive/negative labels in this subset

if nargin == 3
  splitData.labels = labels(ind);
  splitData.data = data(ind,:);
  splitData.pos = sum(splitData.labels);
  splitData.neg = length(splitData.labels)-splitData.pos;
elseif nargin == 2
  splitData.labels = labels(ind);
  splitData.pos = sum(splitData.labels);
  splitData.neg = length(splitData.labels)-splitData.pos;
else
  error('Input Argument Error')
  
end