function [ gw ] = gradient_of_w( M, labels, w, C )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

inner_product = M .* (repmat(w, size(M,1), 1));
product_sum = sum(inner_product, 2);
multiply_label = labels .* product_sum;
denominator = exp(-1*multiply_label);
test = ones(size(denominator)) ./ (denominator + ones(size(denominator)));
total = sum(test - ones(size(test)), 1);

disp(labels.' .* sum(M, 1));
gw = w + C * total * (labels.' .* sum(M, 1));
end

