%% Task 2) 1. Hard margin with linear kernel

%% Initialize
% Load data only once
if exist('train_data', 'var') == 0 || exist('test_data', 'var') == 0
    run('load_data');
end

C = 10^6;
threshold = 0.9;
p = 0;

%% Compute the Kernel
K = get_kernel(train_data, train_data, p);

%% Calculate alpha
size_data = length(train_data(1,:));
alpha = solve_alpha(size_data, train_label, K, C);

%% Calculate b
b = solve_b0(train_label, alpha, K, C, threshold);

%% Calculate g(x) for training data
train_g = get_g(size_data, train_label, alpha, b, K);
% Accuracy
train_acccuracy = mean(sign(train_g) == train_label);

%% Compute the Kernel for test set
K = get_kernel(test_data, train_data, p);

size_data = length(test_data(1,:));
%% Calculate g(x) for test data
test_g = get_g(size_data, train_label, alpha, b, K);
% Accuracy
test_accuracy = mean(sign(test_g) == test_label);

disp(['Training accuracy: ', num2str(train_acccuracy)]);
disp(['Test accuracy: ', num2str(test_accuracy)]);