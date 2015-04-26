function label = predictNN(weights,data)
    sigmoid = @(z) 1./(1+exp(-z));
    [m,n] = size(data);
    n_in = m;                            
    n_out = 2;
    
    n_layer = length(weights);
    
    x = [ones(m,1), data];
    for i = 1:n_layer-1
        x = [ones(m,1),tanh( x * weights{i})];
    end
    
    label = sigmoid(x * weights{n_layer});
    [~,I] = max(label,[],2);
    label = I-1;
   
end
    
    