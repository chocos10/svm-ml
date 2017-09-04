%% Hard margin SVM with a polynomial kernel

%% Initialize
% Load data only once
if exist('train_data', 'var') == 0 || exist('test_data', 'var') == 0
    run('load_data');
end

%% Vector with values of p
p_values = [2 3 4 5 6 7];
% p_values = [6 7 8 9 10 11 12 13 14 15];

C = 10^6;
threshold = 0.9;

% Error rate values
train_acc_hmpk = zeros(length(p_values), 1);
test_acc_hmpk = zeros(length(p_values), 1);

% Make the calculations for every p
for i = 1:length(p_values)
    
    %% Compute the Kernel
    K = get_kernel(train_data, train_data, p_values(i));

    %% Calculate alpha
    size_data = length(train_data(1,:));
    alpha = solve_alpha(size_data, train_label, K, C);

    %% Calculate b
    b = solve_b0(train_label, alpha, K, C, threshold);
    
    %% Calculate g(x) for training data
    train_g = get_g(size_data, train_label,alpha, b, K);
    train_acc_hmpk(i) = mean(sign(train_g) == train_label);
    
    %% Compute the Kernel for test set
    K = get_kernel(test_data, train_data, p_values(i));
    %% Calculate g(x) for test data
    size_data = length(test_data(1,:));
    test_g = get_g(size_data, train_label, alpha, b, K);
    
    test_acc_hmpk(i) = mean(sign(test_g) == test_label);
end

% Plot the error rate against value of p
plot(p_values, train_acc_hmpk, '*-')
hold on
plot(p_values, test_acc_hmpk, '*-')
title('Accuracy of SVM with hard margin and polynomial kernel')
ylim([0.8 1.1])
xlabel('value of p')
ylabel('accuracy')
legend('Training set', 'Test set')
