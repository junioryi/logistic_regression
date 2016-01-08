function [ gw ] = gradient_of_w( M, labels, w, w_dot_x, C )
% Compute the gradient of cost function.
%   grad(f) = w + C * sum_i( 1/(1+exp(-ywx)) - 1) y_i * x

%disp(w_dot_x);
%disp(labels);
multiply_label = labels .* w_dot_x;
denominator = exp(-1*multiply_label) + ones(size(multiply_label));
left_sum  = denominator.^-1 - ones(size(denominator));
total_sum = repmat(left_sum .* labels, 1, size(M, 2)) .* M;
gw = w + C * sum( total_sum );
end





