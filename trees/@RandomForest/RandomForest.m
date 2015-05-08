classdef RandomForest < handle
  
  properties
    NumTrees = 5;
    NumFeatures = 61; %ceil(sqrt(3800))
    Trees = {};
  end
  
  methods
    function newForest = RandomForest(numTrees,numFeatures,max_depth,feature_vector)
      %Constructor for Forest with inputs that govern the number
      %of trees in the forest, the size of the subset of features
      %to use for splits, as well as the size of the subset of
      %data given to each tree for training
      %feature_vector is a vector of all possible features that
      %can be used for splitting
      newForest.NumTrees = numTrees;
      newForest.NumFeatures = numFeatures;
      for i = 1:numTrees
        newTree = DecisionTree(feature_vector,max_depth);
        newForest.Trees{i} = newTree;
      end
    end
    function forest = train(forest,data,labels)
      % Trains each tree in the forest
      [n d] = size(data);
      
      for i = 1:forest.NumTrees
        ind = randi(n,1,n);
        train_labels = double(labels(ind));
        train_data = double(data(ind,:));
        tic
        forest.Trees{i}.train(train_data,train_labels,forest.NumFeatures);
        toc
        disp(sprintf('Trained %d out of %d trees',i,forest.NumTrees))
      end
      
    end
    
    function predictions = predict(forest,test_data)
      %Collects the votes from the Ents in the forest and then
      %give one final decision
      M = forest.NumTrees;
      [n,~] = size(test_data);
      ballot = zeros(n,M);
      for i = 1:M
        ballot(:,i) = forest.Trees{i}.predict(test_data);
        disp(sprintf('Collected %d out of %d votes',i,M))
      end
      predictions = mode(ballot,2);
    end
  end
  
end