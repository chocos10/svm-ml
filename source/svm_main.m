%% Task 3
%% Debug
% Check eval.mat exists before loading it
if exist('eval.mat', 'file') == 0
    error(['Error: svm_main -- File eval.mat could not be found. Please copy the ', ...
        'file into current directory and run the script again.']);
end
if exist('train.mat', 'file') == 0
    error(['Error: svm_main -- File train.mat could not be found. Please copy the ', ...
        'file into current directory and run the script again.']);
end
disp('Files found - eval.mat, train.mat');

%% Initialize
% Load data only once
%% Load train data
load 'train.mat';
train_data = data;
train_label = label;
train_data = norm(train_data);

%% Load eval data
load 'eval.mat';
%% Debug
if exist('evaldata', 'var') == 1
    eval_data = evaldata;
elseif exist('data', 'var') == 1
    eval_data = data;
    eval_label = label;
end
eval_data = norm(eval_data);

%% Initialize defaults

p = 11;
C = 10^6;
threshold = 0.9;

%% Compute the Kernel
K = get_kernel(train_data, train_data, p);

%% Calculate alpha
size_data = length(train_data(1,:));
alpha = solve_alpha(size_data, train_label, K, C);

%% Calculate b
b = solve_b0(train_label, alpha, K, C, threshold);

%% Calculate g(x) for training data
train_g = get_g(size_data, train_label, alpha, b, K);
train_accuracy = mean(sign(train_g) == train_label);

%% Evaluate
%% Compute the Kernel for test set
K = get_kernel(eval_data, train_data, p);

size_data = length(eval_data(1,:));
%% Calculate g(x) for test data
evallabel = get_g(size_data, train_label, alpha, b, K);

%% Debug
if exist('eval_label', 'var') == 1
    eval_accuracy = mean(sign(evallabel) == eval_label);
end

%% Display output message
disp('Workspace variable created - evallabel (100 x 1 column vector)');

%% Normalization function
function [ out ] = norm( data )
% Normalize the data
% Rescale the feature values into the range of [-1 1]
% B = +1, M = -1
    tmp = data';

    mean_tmp = mean(tmp);
    tmp = bsxfun(@minus, tmp, mean_tmp);
    tmp = bsxfun(@rdivide, tmp, std(tmp));

    out = tmp';
end
