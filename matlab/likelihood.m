function [ possibility ] = likelihood( labels, wx )
%% Find cost value 
%%   f(w) = 0.5*w.T*w + C*sum(log(1+exp(-ywx)))

denominator = ones(size(wx)) + exp(-wx .* labels);
possibility = prod(denominator.^-1);
disp(possibility);

end

