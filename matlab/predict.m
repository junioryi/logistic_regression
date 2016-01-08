function [ prediction ] = predict( x, weights )
%% Find cost value 
%%   f(w) = 0.5*w.T*w + C*sum(log(1+exp(-ywx)))

pos_class  = (ones(size(weights)) + exp(-weights)).^-1;
neg_class = (ones(size(weights)) + exp(weights)).^-1;

prediction = bsxfun(@ge, pos_class, neg_class);
end

