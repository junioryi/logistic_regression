fileID = fopen('output', 'w');

[label_vector, instance_matrix] = libsvmread('simple_data');


formatSpec = 'Instance label: %d, features: \n';
out = sprintf(formatSpec, label_vector);
fwrite(fileID, out);
fclose(fileID);