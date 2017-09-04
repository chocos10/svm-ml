function [ b0 ] = solve_b0( desired, alpha, K, C, threshold )
%SOLVE_B0 Summary of this function goes here
%   Calculate b0 for SVM
%   desired : Label vector
%   alpha : value of alpha calculated using solve_alpha
%   K : Kernel matrix
%   C : 10^6 for hard margin, less otherwise
%   threshold : value between 0 and 1 to define the support vectors chosen
% Returns :
%   b0 : value of b0

    sv_index = find(alpha > threshold * max(alpha));

    if C >= 10^6 % Hard margin
        [~, sv_index_max] = max(alpha);
        b0 = desired(sv_index_max) - (alpha.*desired)' * ...
                            K(sv_index_max,:)';
    else % Soft margin
        b = zeros(length(sv_index),1);
        for i = 1:length(b)
            b(i) = desired(sv_index(i)) - (alpha.*desired)' * ...
                            K(sv_index(i),:)';
        end
        b0 = mean(b);  
    end

end

