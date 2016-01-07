fileID = fopen('output', 'w');

data_file = '../data/simple_data'
[label_vector, instance_matrix] = libsvmread( data_file );

% To display full matrix
%{ 
A = full(instance_matrix);
disp(A);
%}

C = 0.1;
num_feature = size(instance_matrix, 1);
w = zeros(1, num_feature);
w(1) = 1;
w(2) = 1;
[ gw ] = gradient_of_w(instance_matrix, label_vector, w, C);

%formatSpec = 'Instance label: %d, features: \n';
%out = sprintf(formatSpec, label_vector);
%fwrite(fileID, out);
fclose(fileID);