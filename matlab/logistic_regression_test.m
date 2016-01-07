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
w_dot_x = sum( instance_matrix .* (repmat(w, col_num, 1)), 2 );

% Compute the gradient of f
[ gw   ] = gradient_of_w(instance_matrix, label_vector, w, w_dot_x, C);
[ cost ] = cost_func(w, C, w_dot_x, label_vector);
[ ak   ] = find_ak(w, gw, eta);

fclose(fileID);





