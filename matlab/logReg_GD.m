function [ w, e, outter_iter ] = logReg_Newton(x, y, C, eps, ksi, eta)
	fileID = fopen('lr_gd.out', 'w');
	formatSpec = 'Iteration: %d, accuracy= %f, f= %f, alpha: %f, time: %f\n';
	n = size(y, 1);
	l = size(x, 2);
	Y = spdiags(y, 0, n, n);
	X = Y*x;
	norm_g0 = norm(C * sum( -0.5 * X ));
	w = zeros(l, 1);

	outter_iter = 1;
	t = cputime;
	while true,
		%fprintf('Iteration: %d, ', outter_iter);
		
		wxs = X * w;
		exp_wxs = exp(-wxs);
		Pminus1 = exp_wxs ./ (1+exp_wxs);
		pyx_left = spdiags(Pminus1, 0, n, n);
		%hessianD = Pminus1./ (1+exp_wxs);
		%D = spdiags(hessianD, 0, n, n);

		% Compute gradient of w
		gw  = w + C * sum(pyx_left * -X)';
		normgw  = norm(gw);
		fval = 0.5 * norm(w)^2 + C * sum(log( 1 + exp(-wxs)));

		if (normgw <= eps * norm_g0)
			break;
		end
	
		%{
		% Conjugate gradient
		s = zeros(l, 1);
		r = -gw;
		d = r;
		normsq_gw = normgw^2;
		normsq_r  = norm(r)^2;
		inner_iter = 1;

		while ~( normsq_r <= ksi^2 * normsq_gw ),
			hessian_d = d + C * ((D*(X*d))' * X)';
			alpha = normsq_r / (d' * hessian_d);
			pre_norm = normsq_r;
			s = s + alpha * d;
			r = r - alpha * hessian_d;
			normsq_r = norm(r)^2;
			beta = normsq_r / pre_norm;
			d = r + beta * d;
			inner_iter = inner_iter + 1;
		end
		%}

		%fprintf('# inner iteration: %d, ', inner_iter);

		% Line search
		s = -gw;
		ak = 1;
		s_x = X * s;
		gw_s = eta * gw' * s;
		cnt = 1;
		while true
			fline = 0.5 * norm(w + ak*s)^2 + C * sum(log(1+exp(-wxs - ak*s_x)));
			if fline <= fval + ak * gw_s
				break
			end
			ak = ak/2;
			cnt = cnt + 1;
		end
		
		% Update 
		w = w + ak * s;
		%fprintf('\n');
		e = cputime - t;

		predict = sign(x * w);
		accuracy = sum(predict == y) / n;

		fprintf(fileID, formatSpec, outter_iter, accuracy, fval, ak, e);
		outter_iter = outter_iter + 1;
	end
	e = cputime - t;
	fclose(fileID);







