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
lr=1e-3;
w = zeros(n_vars, 1); 
k_array= [-8,-6,-4,-2,0,2];
iter = 0;                            % iteration counter
max_iter = 1000;                     % maximum iteration

train_accuracy = zeros(length(k_array),1);
test_accuracy = zeros(length(k_array),1);

printf("Non-regularized:\n");
for k=1:length(k_array)
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
    y(exp_theta == Inf) = 1;
   
    grad = ((X_tr'*(y_tr - y))- (pow2(k_array(k))*w)); 
      
    % take a step using the learning rate
    w_new = w + (lr.*grad); 

    %printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));
    %fflush(stdout);

    % stopping criteria and perform update if not stopping
   if mean(abs(grad)) < tolerance
       w = w_new;
        break;
    else
        w = w_new;
    end
    
    if iter >= max_iter 
      break;
    end
  endwhile      
    % use w for prediction
    % calculate testing accuracy
   
    pred_test = zeros(n_ts, 1);      % initialize prediction vector
    pred_test = (exp(X_ts*w)./(1+exp(X_ts*w)));
    pred_test = pred_test >= 0.5;
    
    test_accuracy(k) = mean(pred_test == y_ts);
    printf('For k = %d\ntest accuracy = %d\n',pow2(k_array(k)),test_accuracy(k));

  % repeat the similar prediction procedure to get training accuracy
    pred_train = zeros(n_tr, 1);               % initialize prediction vector
    pred_train = (exp(X_tr*w)./(1+exp(X_tr*w)));
    pred_train = pred_train >= 0.5;
    train_accuracy(k) = mean(pred_train == y_tr);
    printf('train accuracy = %d\n', train_accuracy(k));
    
 endfor   
    
    title(sprintf("L2 Regularization"));
    hold on;
    xlabel('K');
    ylabel('Accuracy');
    plot(k_array, test_accuracy*100.0);
    plot(k_array, train_accuracy*100.0);
    legend('Test Accuracy','Train Accuracy');
    hold off;
 
% Non-regularized
w = zeros(n_vars, 1); 
iter = 0;                            % iteration counter
max_iter = 1000;                     % maximum iteration
w_new = zeros(n_vars,1);

while true
    iter = iter + 1;                 % start iteration
    % calculate gradient
    grad = zeros(n_vars, 1);         % initialize gradient
    % compute the gradient with respect to w_j here  
    exp_theta = exp(X_tr*w);
    y = exp_theta./(1+exp_theta);
    y(exp_theta == Inf) = 1;
   
    grad += X_tr'*(y_tr - y); 
      
    % take a step using the learning rate
    w_new = w + lr.*grad; 

    %printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));
    %fflush(stdout);

    % stopping criteria and perform update if not stopping
   if mean(abs(grad)) < tolerance
       w = w_new;
        break;
    else
        w = w_new;
    end
    
    if iter >= max_iter 
      break;
    end
endwhile      
    % use w for prediction
    % calculate testing accuracy
    printf("\nRegularized:\n");  
    train_acc_reg = 0;
    test_acc_reg = 0; 
    pred_test = zeros(n_ts, 1);      % initialize prediction vector
    pred_test = (exp(X_ts*w)./(1+exp(X_ts*w)));
    pred_test = pred_test >= 0.5;
    
    test_acc_reg = mean(pred_test == y_ts);
    printf('test accuracy = %d\n',test_acc_reg);

  % repeat the similar prediction procedure to get training accuracy
    pred_train = zeros(n_tr, 1);               % initialize prediction vector
    pred_train = (exp(X_tr*w)./(1+exp(X_tr*w)));
    pred_train = pred_train >= 0.5;
    train_acc_reg = mean(pred_train == y_tr);
    printf('train accuracy = %d\n', train_acc_reg); 

printf("\n\n\t\tTrain accuracy\tTest accuracy\n");
printf("Regularized\t%d\t%d\n",train_accuracy(3),test_accuracy(3));
printf("Non-regularized\t%d\t%d\n",train_acc_reg,test_acc_reg);



	