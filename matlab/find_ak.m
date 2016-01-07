function [ a_k, new_weight ] = find_ak(x, w, weights, grad_w, old_cost, C, eta, labels )
%% Find a_k to update w
%%  f(w^k + a_k * s^k) <= 
%%      f(w^k) + eta * a_k grad(f(w^k)).T * s^k

a_k = 2.0;
found = false;
num_col = size(x, 1);
while ~found
    a_k = a_k / 2;
    % new_weight: w.T * x + a_k * d.T * x
    new_w = w - a_k * grad_w;
    new_weight = weights + a_k * sum(x .* repmat(-grad_w, num_col, 1), 2); 
    new_cost = cost_func(new_w, C, new_weight, labels);

    if new_cost <= old_cost - eta*a_k*dot(grad_w, grad_w)
        found = true;
    elseif a_k < 0.5^10
        found = true;
        disp('line search fail')
    end
end

