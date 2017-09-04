function [ alpha ] = solve_alpha( size_data, desired, K, C )
%SOLVE_ALPHA
% Calculates the value of alpha for SVM
%   size_data : Number of samples
%   desired : Label vector
%   K : Kernel matrix
%   C : 10^6 for hard margin, less otherwise
% Returns :
%   alpha : alpha values given by the optimisation problem


    %% Initialize inputs for quadprog

    H = zeros(size_data, size_data);
    for i = 1:size_data
        for j = 1:size_data
            H(i,j) = desired(i) * desired(j) * K(i,j);
        end
    end
    f = -ones(size_data, 1);
    Aeq = desired';
    Beq = 0;
    lb = zeros(size_data, 1);
    ub = ones(size_data, 1) * C;
    x0 = [];
    options = optimset('LargeScale', 'off', 'MaxIter', 1000);
    A = [];
    b = [];
    
    %% Call quadprog
    [ alpha, ~, ~ ] = quadprog(H, f, A, b, Aeq, Beq, ...
                        lb, ub, x0, options);

end


