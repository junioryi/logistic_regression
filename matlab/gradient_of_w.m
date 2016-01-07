function [ gw ] = gradient_of_w( M, labels, w, w_dot_x, C )
% Compute the gradient of cost function.
%   grad(f) = w + C * sum_i( 1/(1+exp(-ywx)) - 1) y_i * x

multiply_label = labels .* w_dot_x;
denominator = exp(-1*multiply_label) + ones(size(multiply_label));
output_size = size(denominator);

left_sum  = ones(output_size) ./ denominator - ones(output_size);
right_sum = repmat(labels .* left_sum, 1, size(M, 2)) .* M;

gw = w + C * sum( right_sum );
end





