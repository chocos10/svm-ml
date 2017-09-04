% Soft margin SVM with a polynomial kernel

%% Initialize
% Load data only once
if exist('train_data', 'var') == 0 || exist('test_data', 'var') == 0
    run('load_data');
end

% Vector with values of p
p_values = [2 3 4 5 6 7];
% p_values = [6 7 8 9 10 11 12 13 14 15];

% Vector with values of C
C_values = [0.1 0.6 1.1 2.1];

% Error rate values
train_acc_smpk = zeros(length(p_values), length(C_values));
test_acc_smpk = zeros(length(p_values), length(C_values));

%% Make the calculations for every p
for i = 1:length(p_values)
    for j = 1:length(C_values)
        %% Compute the Kernel
        K = get_kernel(train_data, train_data, p_values(i));

        %% Calculate alpha
        size_data = length(train_data(1,:));
        alpha = solve_alpha(size_data, train_label, K, C_values(j));

        %% Calculate b
        b = solve_b0(train_label, alpha, K, C_values(j), threshold);

        %% Calculate g(x) for training data
        train_g = get_g(size_data, train_label,alpha, b, K);
        train_acc_smpk(i,j) = mean(sign(train_g) == train_label);
        
        %% Compute the Kernel for test set
        K = get_kernel(test_data, train_data, p_values(i));
        
        %% Calculate g(x) for test data
        size_data = length(test_data(1,:));
        test_g = get_g(size_data, train_label, alpha, b, K);
        test_acc_smpk(i,j) = mean(sign(test_g) == test_label);
    end
end


%% Plot the results
figure
% colormap spring
surf(p_values, C_values, train_acc_smpk')
hold on
surf(p_values, C_values, test_acc_smpk')
xlabel('p')
ylabel('C')
zlabel('accuracy')
title('SVM accuracy with soft margin and polynomial kernel, against p and C values')
legend('Training set', 'Test set')

