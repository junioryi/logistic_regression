function [ gw ] = gradient_of_w( M, labels, w, w_dot_x, C )
% Compute the gradient of cost function.
%   grad(f) = w + C * sum_i( 1/(1+exp(-ywx)) - 1) y_i * x

multiply_label = labels .* w_dot_x;

%denominator = exp(-1*multiply_label) + ones(size(multiply_label));
denominator = bsxfun(@plus, exp(-1*multiply_label), 1);

%left_sum  = denominator.^-1 - ones(size(denominator));
left_sum = bsxfun(@minus, denominator.^-1, 1);

%total_sum = repmat(left_sum .* labels, 1, size(M, 2)) .* M;
total_sum = bsxfun(@times, M, left_sum .* labels);
gw = w + C * sum( total_sum );
end





