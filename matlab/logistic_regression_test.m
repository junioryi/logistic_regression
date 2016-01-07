fileID = fopen('output', 'w');

% Variables
data_file = '../data/simple_data';
C   = 0.1;
eta = 0.1;

% Read sparse matrix format
[label_vector, instance_matrix] = libsvmread( data_file );

num_feature = size(instance_matrix, 1);
w = zeros(1, num_feature);
w(1) = 1;
w(2) = 1;

[ col_num, row_num ] = size( instance_matrix );
% Compute the w.T dot x, will use several time later.
weights = sum( instance_matrix .* (repmat(w, col_num, 1)), 2 );

% Compute the gradient of f.
[ gw   ] = gradient_of_w(instance_matrix, label_vector, w, weights, C);
% Compute cost value given w.
[ cost ] = cost_func(w, C, weights, label_vector);
% Find ak using line search.
[ ak   ] = find_ak(instance_matrix, w, weights, gw, cost, C, eta, label_vector);

fclose(fileID);





