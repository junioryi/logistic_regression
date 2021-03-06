fileID = fopen('output', 'w');

% Variables
%data_file = '../data/simple_data';
data_file = '/tmp2/r03222055/kddb';
eta     = 0.01;
C       = 0.1;
xi      = 0.1;
epsilon = 0.01; % Stopping Condition

% Read sparse matrix format
disp('start reading data');
[ label_vector, instance_matrix ] = libsvmread( data_file );
disp('finish reading data');
label_vector(label_vector == 0) = -1;

[ row_num, col_num ] = size( instance_matrix );
w = zeros(1, col_num);
% Compute the w.T dot x, will use it several time later.
%weights = sum( instance_matrix .* (repmat(w, row_num, 1)), 2);
weights = sum( bsxfun(@times, instance_matrix, w), 2);

% For stopping condition
[ gw0 ] = gradient_of_w(instance_matrix, label_vector, w, weights, C);
norm_gw0 = norm(gw0);
formatSpec = 'Iteration: %d, objective function value: %f, time: %f\n';

t = cputime;

for i = 1:2

	% Compute the gradient of f.
	[ gw   ] = gradient_of_w(instance_matrix, label_vector, w, weights, C);
	[ sk   ] = conjugate_gradient(instance_matrix, gw, weights, label_vector, C, xi);
	% Compute cost value given w.
	[ cost ] = cost_func(w, C, weights, label_vector);
	% Find ak using line search.
	[ ak, new_weights ] = line_search(instance_matrix, w, weights, sk, cost, C, eta, label_vector);
	% Update w
	w = w - ak * sk;
	weights = new_weights;

	e = cputime - t;

	msg = sprintf(formatSpec, i, cost, e);
	disp(msg);
	fprintf(fileID, formatSpec, i, cost, e);

	if norm(gw) <= epsilon * norm_gw0 
		disp('Meet stopping condition.');
		break;
	end
end

[ predictions ] = predict(instance_matrix, weights);
label_vector(label_vector==-1) = 0;
correct = bsxfun(@eq, label_vector, predictions);
percent = sum(correct);
accuracy = percent / (1.0 * size(correct, 1));
disp('Accuracy:');
disp(accuracy);

fclose(fileID);
exit;





