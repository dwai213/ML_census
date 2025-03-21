function [W,w] = train_mulNN(data,labels,params)        
close all
[TrainNum,FeatNum] = size(data);
sigmoid = @(z) 1./(1+exp(-z));          % Last layer activate sigmoid function
lambda = params{5};                     % Penalty parameter
maxiter = params{3};                    % Maximum iteration
fnc = params{2};                        % Error Funciton selection
eta = params{1};                        % Learning Rate

n_in = FeatNum;                         % Number of input feature
n_hid = [];
for hid = 1:length(params{4})           % Number of hidden nodes
    n_hid(hid) = params{4}(hid);
end    
n_out = 2;                              % Number of output classes
n = [n_hid,n_out];

% Function Definition
if strcmp(params{2},'mean_squared')
    error = @(s2,y,w1) 0.5*norm(y - sigmoid(s2))^2 ;%+ lambda*norm(w1,1);                            % Error scalar
    delta2 = @(s2,y) sigmoid(s2).*(1-sigmoid(s2)).*(sigmoid(s2)-y);
    delta1 = @(delta,s1,s2,y,w2,d) sum((repmat(delta,n(d),1)).* w2 .* repmat((1-tanh(s1).^2)',1,n(d+1)),2)' ;
else
    error = @(s2,y,w1) -(y.* log(sigmoid(s2)) + (1-y).*log(1-sigmoid(s2)))*ones(n_out,1);% + lambda*norm(w1,1);
    delta2 = @(s2,y) (-y.*(1-sigmoid(s2))+(1-y).*sigmoid(s2));
    delta1 = @(delta,s1,s2,y,w2,d) sum((repmat(delta,n(d),1)).* w2 .* repmat((1-tanh(s1).^2)',1,n(d+1)),2)';
end


epsilon = 1e-10;                           % Maximum difference of error function for stopping
Err(2) = 0;                                % Initialize error function value
Err(1) = 1;
iter = 1;                                  % Iteration counter
r = 1;                                     % Record ith 1000 iteration
difference = 1;                            % Initialize error function difference value
w = {};                                    % Store weights for every 1000 iteration     

% Initialize weights [-e,e], here [-0.2,0.2]
for i = 1:length(n_hid)
    if i == 1
       W{i} = 2 * 0.1 * rand(n_in + 1, n_hid(i)) - 0.1 * ones(n_in + 1, n_hid(i));
    elseif i<=length(n_hid)
       W{i} = 2 * 0.1 * rand(n_hid(i-1) + 1, n_hid(i)) - 0.1 * ones(n_hid(i-1) + 1, n_hid(i));
    end
end

W{length(n_hid)+1} = 2 * 0.1 * rand(n_hid(end) + 1, n_out) - 0.1 * ones(n_hid(end) + 1, n_out);

  
while difference>epsilon && iter<= maxiter
    
    for i = 1:length(n_hid)                                         % Initialize input layer gradient
        if i == 1
            G{i} = zeros(n_in+1 , n_hid(i));
        elseif i<=length(n_hid)
            G{i} = zeros(n_hid(i-1)+1 , n_hid(i));
        end
    end
    G{length(n_hid)+1} = zeros(n_hid(end) , n_out);                 % Initialize output layer gradient
    
    idx = randi(TrainNum);                                          % pick one data at random
    for j = 1:length(n_hid)                                         % Forward pass compute x^(l) (lth layer output) 
        if j ==1
           s{j} = [1,data(idx,:)] * W{j};
        else
           s{j} = [1,tanh(s{j-1})] * W{j};
    
        end
    end
    s{length(n_hid)+1} = [1,tanh(s{length(n_hid)})] * W{length(n_hid)+1};
    
    for k = length(n_hid)+1 : -1 : 1                               % Backpropogation to compute derivative
        if k == length(n_hid)+1
           Delta{k} = delta2(s{k},labels(idx,:));
           G{k} =  [1 tanh(s{k-1})]' * Delta{k};   
        elseif k == 1
           Delta{k} = delta1(Delta{k+1},s{k},s{k+1},labels(idx,:),W{k+1}(2:end,:),k) ;
           G{k} = [1,data(idx,:)]'* Delta{k}; %+ lambda * sign(W{k});
        else
           Delta{k} = delta1(Delta{k+1},s{k},s{k+1},labels(idx,:),W{k+1}(2:end,:),k);
           G{k} =  [1,s{k-1}]' * Delta{k};
        end
    end
    
    for i = 1:length(n_hid)+1                                      % Stochastic gradient descent weights update
        W{i} = W{i} - eta/sqrt(iter) * G{i};
%         W{i} = W{i}- eta/(floor(iter/1000)+1) * G{i};
    end
    
    if mod(iter,1000) == 0
       w{r} = W;
       r = r+1;
    end
    
    x = [1, data(idx,:)];
    for i = 1:length(n_hid)
        x = [1,tanh( x * W{i})];
    end
    
    output = sigmoid(x * W{length(n_hid)+1});
    Err(2) = error(output,labels(idx,:),W{1});
%     if mod(iter,1000)==0
%         figure(1)
%         plot(iter,Err(2),'x')
%         hold on;
%     end
    difference = abs(Err(2)-Err(1));
    Err(1) = Err(2);
    iter = iter+1;
end
difference
iter

end

