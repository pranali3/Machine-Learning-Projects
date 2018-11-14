% printing option
more off;

% read files
D_tr = csvread('spambasetrain.csv'); 
D_ts = csvread('spambasetest.csv');  

% construct x and y for training and testing
X_tr = D_tr(:, 1:end-1);
y_tr = D_tr(:, end);
X_ts = D_ts(:, 1:end-1);
y_ts = D_ts(:, end);

% number of training / testing samples
n_tr = size(D_tr, 1);
n_ts = size(D_ts, 1);

% add 1 as a feature
X_tr = [ones(n_tr, 1) X_tr];
X_ts = [ones(n_ts, 1) X_ts];

% perform gradient descent :: logistic regression
n_vars = size(X_tr, 2);              % number of variables
tolerance = 1e-2;                    % tolerance for stopping criteria

w = zeros(n_vars, 1); 
lr_array= [1,1e-2,1e-4,1e-6];
iter = 0;                            % iteration counter
max_iter = 1000;                     % maximum iteration

pred_test=zeros(n_ts,1);
pred_train=zeros(n_tr,1);

train_accuracy = zeros(max_iter,1);
test_accuracy = zeros(max_iter,1);

for lr=1:length(lr_array)
  %printf("For learning rate = %d\n", lr_array(lr));
  w = zeros(n_vars, 1);
  iter = 0;
  while true
    iter = iter + 1;                 % start iteration
    % calculate gradient
    grad = zeros(n_vars, 1);         % initialize gradient
    % compute the gradient with respect to w_j here  
    exp_theta = exp(X_tr*w);
    y = exp_theta./(1+exp_theta);
    y(exp_theta == Inf) = 1; % overflow
    y(exp_theta == -Inf) = 0; % underflow
    grad += (X_tr'*(y_tr - y)); 
      
    % take a step using the learning rate
    w_new = w + (lr_array(lr).*grad); 

    %printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));
    %fflush(stdout);

    % stopping criteria and perform update if not stopping
   if mean(abs(grad)) < tolerance
       w = w_new;
        break;
    else
        w = w_new;
    endif
    
    if iter >= max_iter 
      break;
    endif
           
    % use w for prediction
    % calculate testing accuracy
    
      pred_test = (exp(X_ts*w)./(1+exp(X_ts*w)));
      pred_test = pred_test >= 0.5;
    
      test_accuracy(iter) = mean(pred_test == y_ts);
    %printf('test accuracy = %d\n', test_accuracy(iter));

  % repeat the similar prediction procedure to get training accuracy
   % pred_train = zeros(n_tr, 1);               % initialize prediction vector
      pred_train = (exp(X_tr*w)./(1+exp(X_tr*w)));
      pred_train = pred_train >= 0.5;
      train_accuracy(iter) = mean(pred_train == y_tr);
    %printf('train accuracy = %d\n', train_accuracy(iter));
  
  endwhile
   
    subplot(2,2,lr);
    title(sprintf('Learning Rate = %f',lr_array(lr)));
    hold on;
    xlabel('Number of Iterations');
    ylabel('Accuracy');
    plot(1:1000, test_accuracy);
    plot(1:1000, train_accuracy);
    if lr == 1
      axis([1 1000 0.2 0.8]);
    elseif lr == 2
      axis([1 1000 0.8 0.93]);
    else
      axis([1 1000 0.89 0.93]);
    endif
    legend('Test Accuracy','Train Accuracy','Location','southeast');
    hold off; 
endfor




	