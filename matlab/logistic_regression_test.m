fileID = fopen('output', 'w');

% Variables
data_file = '../data/simple_data';
C       = 0.1;
eta     = 0.1;
epsilon = 0.01; % Stopping Condition

% Read sparse matrix format
[label_vector, instance_matrix] = libsvmread( data_file );

num_feature = size(instance_matrix, 1);
w = zeros(1, num_feature);
w(1) = 1;
w(2) = 1;

[ col_num, row_num ] = size( instance_matrix );
% Compute the w.T dot x, will use several time later.
weights = sum( instance_matrix .* (repmat(w, col_num, 1)), 2 );

% For stopping condition
[ gw0 ] = gradient_of_w(instance_matrix, label_vector, w, weights, C);
norm_gw0 = norm(gw0);

for i = 1:100

	% Compute the gradient of f.
	[ gw   ] = gradient_of_w(instance_matrix, label_vector, w, weights, C);
	% Compute cost value given w.
	[ cost ] = cost_func(w, C, weights, label_vector);
	% Find ak using line search.
	[ ak, new_weights ] = find_ak(instance_matrix, w, weights, gw, cost, C, eta, label_vector);
	% Update w
	w = w - ak * gw;
	weights = new_weights;
	disp(cost);

	if norm(gw) <= epsilon * norm_gw0 
		disp(norm(gw));
		disp(norm_gw0);
		break;
	end
end

fclose(fileID);





