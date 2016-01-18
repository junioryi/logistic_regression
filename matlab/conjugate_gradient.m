function [ sk ] = conjugate_gradient( M, gw, weights, labels, C, xi )
% Compute the gradient of cost function.
%   grad(f) = w + C * sum_i( 1/(1+exp(-ywx)) - 1) y_i * x

l = size(weights, 1);
sk = zeros(size(gw, 2), 1);
rk = -gw;
dk = rk;

%Dii = exp( -labels .* weights ) ./ ( ones(size(weights)) + exp(-labels .* weights) ).^2;
exp_ywx = exp(-labels .* weights);
Dii_deno = bsxfun(@plus, exp_ywx, 1);
Dii = bsxfun(@rdivide, exp_ywx, Dii_deno.^2);
D = sparse(1:l, 1:l, Dii, l, l);

MT = transpose(M);

for i = 1:100000
	norm_r = norm(rk);
	if norm_r <= xi * norm(gw)
		break
	end
	dkT = dk.';
	hessian_d = dkT + C*(MT * D * M * dkT);
	alpha = norm_r^2 / (dk * hessian_d);
	sk_1  = sk + alpha * dkT;
	rk_1  = rk - alpha * hessian_d.'; 
	beta  = norm(rk_1)^2 / norm_r^2;
	
	% Update 
	dk = rk_1 + beta * dk;
	sk = sk_1;
	rk = rk_1;
end

sk = -sk.';

end

















