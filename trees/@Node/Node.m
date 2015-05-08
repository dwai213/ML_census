classdef Node < handle
  
  properties
    Name = []; %used only for exporting
    Parent = [];
    Left = [];
    Right = [];
    split_rule = [-1,-1];
    positive = 0;
    negative = 0;
    curr_depth = [];
    max_depth = [];
  end
  
  methods
    function newNode = Node(parent,pos,neg)
      %Constructor for a node within a DecisionTree
      if nargin == 3
        newNode.Parent = parent; %this is a node     
        newNode.positive = pos; %this is numbers of positive labels
        newNode.negative = neg;
        newNode.curr_depth = parent.curr_depth + 1;
        newNode.max_depth = parent.max_depth;
      elseif nargin == 1
        newNode.Parent = parent;
        if strcmp(parent,'root')
          newNode.curr_depth = 0; %assigns the 0 depth to the root node
        end
      else
        error('Improper Number of Arguments')
      end
    end
    
    function grownNode = grow(node,data,labels,numFeatures)
      %Learning/Growing the Decision Tree
      if (length(unique(labels)) == 1)
        %Returns a Leaf node if the node is pure enough
        grownNode = node;
        if labels(1) == 1
          grownNode.positive = sum(labels);
        elseif labels(1) == 0
          grownNode.negative = length(labels) - sum(labels);
        end
      elseif node.curr_depth >= node.max_depth
        %Returns a Leaf node if the you have reached/exceeded the
        %max depth
        grownNode = node;
        if labels(1) == 1
          grownNode.positive = sum(labels);
        elseif labels(1) == 0
          grownNode.negative = length(labels) - sum(labels);
        end        
      else
        %Otherwise, further split the data into purer nodes and
        %recursively call grow again
        if nargin == 3
          [split threshold] = node.segmentor(data,labels);
        elseif nargin == 4
          [split threshold] = node.segmentor(data,labels,numFeatures);
        else
          error('Incorrect # of Inputs');
        end
        node.split_rule = [split,threshold];
        feature = data(:,split);
        left_ind = (feature <= threshold);
        right_ind = (feature > threshold);
        
        left = splitter(left_ind,labels,data); %gives the subset of data/labels that belongs to the left node
        right = splitter(right_ind,labels,data);
        
        if (isempty(left.labels) || isempty(right.labels))
          %Captures edge case where optimal split doesn't improve the
          %children nodes' purity. Therefore, return as a leaf node
          grownNode = node;
          grownNode.positive = sum(labels);
          grownNode.negative = length(labels) - sum(labels);
        else
          %Keep on growing descendents
          node.Left = Node(node,left.pos,left.neg).grow(left.data,left.labels);
          node.Right = Node(node,right.pos,right.neg).grow(right.data,right.labels);
          grownNode = node;
        end
      end
      
    end
    
    function [split boundary] = segmentor(node,data,labels,numFeatures)
      %Brute force attempts various splits with Information Gain
      %as a metric. The split that provides the best Information
      %Gain is chosen and returned
      [n,d] = size(data);
      sums_1 = labels*data;
      sums_0 = ones(1,n)*data - sums_1;
      means_1 = sums_1/sum(labels);
      means_0 = sums_0/(n-sum(labels));
      thresholds = .5*(means_1+means_0);
      
      parent_entropy = entropy(node.positive,node.negative);
      if nargin == 4
        ind = randperm(d,numFeatures);
        feature_cnts = numFeatures;
      elseif nargin == 3
        feature_cnts = d;
      else
        error('Incorrect # of Inputs')
      end
      gains = zeros(1,feature_cnts);
      for i = 1:feature_cnts
        if nargin == 4
          j = ind(i);
          thresh = thresholds(j);
          feature = data(:,j);
        elseif nargin == 3
          thresh = thresholds(i);
          feature = data(:,i);
        else error('Incorrect # of Inputs');
        end
        
        left_ind = (feature <= thresh);
        right_ind = (feature > thresh);
        
        left = splitter(left_ind,labels);
        right = splitter(right_ind,labels);
        
        left_entropy = entropy(left.pos,left.neg);
        right_entropy = entropy(right.pos,right.neg);
        avg = (length(left.labels)*left_entropy + length(right.labels)*right_entropy)/length(labels);
        
        gains(i) = parent_entropy - avg;
      end
      [~,split] = max(gains);
      boundary = thresholds(split);
    end
    
    function isLeaf = isLeaf(node)
      %Returns true if node is a Leaf Node (no more children)
      if isempty(node.Left) && isempty(node.Right)
        isLeaf = 1;
      else
        isLeaf = 0;
      end
    end
    
    function label = nodeLabel(node)
      %Returns the label of the node, determined by the mode of
      %the data sorted into this node
      a = ones(1,node.positive);
      b = zeros(1,node.negative);
      label = mode([a b]);
    end
    
  end
  
end

