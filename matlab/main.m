fileID = fopen('output', 'w');

data_file = '../data/simple_data';
%data_file = '/tmp2/r03222055/kddb';
eta = 0.01;
C   = 0.1;
ksi = 0.1;
eps = 0.01;

%fprintf('\nStart reading data...\n');
[ y, x ] = libsvmread( data_file );
%fprintf('Finish reading data.\n');
y = 2*y-1;

t = cputime;
[ w, t, n ] = logReg_Newton(x, y, C, eps, ksi, eta);
e1 = cputime - t;
fprintf(fileID, 'Newton done.\n  Total iteration: %d, time: %f\n', n, t);
[ w, t, n ] = logReg_GD(x, y, C, eps, ksi, eta);
e2 = cputime - e1; 
fprintf(fileID, 'Gradient Descent done.\n  Total iteration: %d, time: %f\n', n, t);
exit;

