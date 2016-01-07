fileID = fopen('output', 'w');

[label_vector, instance_matrix] = libsvmread('simple_data');

% To display full matrix
%{ 
A = full(instance_matrix);
disp(A);
%}

C = 0.1;
num_feature = size(instance_matrix, 1)
w = zeros(1, num_feature);
[ gw ] = gradient_of_w(instance_matrix, label_vector, w, C);
disp(gw);

%formatSpec = 'Instance label: %d, features: \n';
%out = sprintf(formatSpec, label_vector);
%fwrite(fileID, out);
fclose(fileID);