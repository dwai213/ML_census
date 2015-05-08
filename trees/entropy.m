function val = entropy( positive, negative)
% Returns the entropy of the a node with the inputted amount of
% positive and negative labels

N = positive + negative;
if N == 0 || positive == 0 || negative == 0
  val = 0;
else
  val = -( positive/N*log2(positive/N) + negative/N*log2(negative/N));
end

end

