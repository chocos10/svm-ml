function [ g ] = get_g(size_data, train_label, alpha, b0, K)
%GET_G
% Evaluates the discriminant function g(.) for each sample of a dataset :

g = sum(bsxfun(@times, K, (alpha .* train_label)') , 2) + b0 * ...
                ones(size_data, 1);

end

