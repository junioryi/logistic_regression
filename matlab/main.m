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

[ w, t, n ] = logReg_Newton(x, y, C, eps, ksi, eta);
%fprintf('Total iteration: %d, total time: %f\n', n, t);

