classdef DecisionTree
  
  properties
    numNodes = 1;
    features = [];
    root = [];
    max_depth = 20;
  end
  
  methods
    function newTree = DecisionTree(features,max_depth)
      %Constructor for a DecisionTree. Includes the features in
      %the design matrix used for decision making
      newTree.features = features;
      newTree.root = Node('root');
      newTree.max_depth = max_depth;
      newTree.root.max_depth = max_depth;
    end
    
    function tree = train(tree,data,label,numFeatures)
      %Given a blank tree, returns a decision tree based on
      %optimal information gain splitting
      tree.root.positive = sum(label);
      tree.root.negative = length(label)-tree.root.positive;
      if nargin == 3
        tree.root = tree.root.grow(data,label);        
      elseif nargin == 4
        tree.root = tree.root.grow(data,label,numFeatures);
      else
        error('Incorrect # of Inputs');
      end
      tree.numNodes = tree.inspect();
    end
    
    function predictions = predict(tree,test_data)
      % Given a trained tree, gives the predictions for inputted
      % test_data
      [n,d] = size(test_data);
      predictions = zeros(n,1);
      for i = 1:n
        leaf = 0;
        curr_node = tree.root;
        while leaf ~= 1
          if isLeaf(curr_node)
            predictions(i) = nodeLabel(curr_node);
            leaf = 1;
          else
            split_rule = curr_node.split_rule;
            col = split_rule(1); thresh = split_rule(2);
            if test_data(i,col) <= thresh
              curr_node = curr_node.Left;
            elseif test_data(i,col) > thresh
              curr_node = curr_node.Right;
            else
              error('Cannot classify data')
            end
          end
        end
      end
      
    end
    
    function splits = predictVerbose(tree,data)
      % Returns the list of splits used to sort the data in the
      % given trained DecisionTree, tree
      leaf = 0;
      curr_node = tree.root;
      splits = {};
      while leaf ~= 1
        if isLeaf(curr_node)
          splits = [splits {nodeLabel(curr_node)}];
          leaf = 1;
        else
          split_rule = curr_node.split_rule;
          col = split_rule(1); thresh = split_rule(2);
          if data(col) <= thresh
            curr_node = curr_node.Left;
            split = sprintf('%s <= %3.2f',tree.features{col},thresh);
          elseif data(col) > thresh
            curr_node = curr_node.Right;
            split = sprintf('%s > %3.2f',tree.features{col},thresh);
          else
            error('Cannot classify data')
          end
          splits = [splits {split}];
        end
      end
      
    end
    
    function numNodes = inspect(tree)
      %Returns the number of nodes in the given tree
      
      curr_node = tree.root;
      frontier = {curr_node};
      numNodes = 0;
      while ~isempty(frontier)
        frontier = frontier(2:end);
        numNodes = numNodes + 1;
        if ~isempty(curr_node.Left)
          frontier = [frontier {curr_node.Left}];
        end
        if ~isempty(curr_node.Right)
          frontier = [frontier {curr_node.Right}];
        end
        if isempty(frontier)
          continue
        else
          curr_node = frontier{1};
        end
      end
    end
    
    function export(tree)
      %Goes through the entire tree and spits out a .dot file
      %that can be then used to visualize the tree
      
      f = fopen('tree.dot','w');
      fprintf(f,'digraph d {\n');
      fprintf(f,'nodesep=0.2\n');
      fprintf(f,'node [color=Blue,fontname=Arial,shape=box]\n');
      fprintf(f,'edge [color=Turquoise, style=dashed]\n');
      
      curr_node = tree.root;
      frontier = {curr_node};
      numNodes = 0;
      while ~isempty(frontier)
        %graphviz building code
        rule = frontier{1}.split_rule;
        if curr_node.isLeaf()
          fprintf(f,'x%d[label="pos: %d\\n neg: %d"]\n',...
            numNodes,curr_node.positive,curr_node.negative);
        else
          fprintf(f,'x%d[label="''%s'' <= %3.2f \\n pos: %d\\nneg: %d"]\n',...
            numNodes,tree.features{rule(1)},rule(2),curr_node.positive,curr_node.negative);
        end
        curr_node.Name = numNodes;
        
        if ~isempty(curr_node.Parent) && ~strcmp(curr_node.Parent,'root')
          fprintf(f,'x%d -> x%d \n',curr_node.Parent.Name,curr_node.Name);
        end
        
        %tree searching code
        frontier = frontier(2:end);
        numNodes = numNodes + 1;
        if ~isempty(curr_node.Left)
          frontier = [frontier {curr_node.Left}];
        end
        if ~isempty(curr_node.Right)
          frontier = [frontier {curr_node.Right}];
        end
        
        if isempty(frontier)
          continue
        else
          curr_node = frontier{1};
        end
      end
      
      fprintf(f,'\n}');
      fclose(f);
      
    end
  end
  
end