clear
close all 

%% Load train data
load 'train.mat';
train_data = data;
train_label = label;

%% Load test data
load 'test.mat';
test_data = data;
test_label = label;

%% Free up memory
clear data;
clear label;

%% Normalize data between -1 and 1 
train_data = norm(train_data);
test_data = norm(test_data);

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