fileID = fopen('output', 'w');

data_file = '../data/simple_data';
eta = 0.01;
C   = 0.1;
ksi = 0.1;
eps = 0.01;

fprintf('\nStart reading data...\n');
[ y, x ] = libsvmread( data_file );
fprintf('Finish reading data.\n');
y = 2*y-1;

[ w ] = logReg_Newton(x, y, C, eps, ksi, eta);
[ w ]  = lr(y, x, C, eps, ksi, eta);

