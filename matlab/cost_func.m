function [ cost ] = cost_func( w, C, w_x, labels )
%% Find cost value 
%%   f(w) = 0.5*w.T*w + C*sum(log(1+exp(-ywx)))

right = C * sum(log( ones(size(labels)) + exp(-1.0 * w_x .* labels) ));
cost = 0.5 * dot(w, w) + right;

end

