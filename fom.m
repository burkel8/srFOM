% FUNCTION: out = FOM(A,b,param)
% A function which computes the standard Arnoldi approximation to f(A)b

% INPUT:  A   The matrix
%         b   The vector
%        param An input struct with the following fields
%
%        param.max_it   The maximum number of Arnoldi iterations
%        param.n        The dimension of A
%        param.reorth   Re-orthogonalization parameter
%        param.fm       fm = @(A,b) f(A)*b
%        param.tol      Convergence tolerance

% OUTPUT: out           An output struct with the following fields
%
%         out.m         Number of Arnoldi iterations executed
%         out.approx    Approximation to f(A)b
%         out.err       Vector storing exact relative errors at each
%                       iteration
function out = fom(A,b,param)
max_it = param.max_it;
n = param.n;
fm = param.fm;
reorth = param.reorth;
exact = param.exact;
tol = param.tol;
d = param.d;
err_monitor = param.err_monitor;

mv = 0;
d_it = 1;

prev_approx = b;

V = zeros(n,max_it);
H = zeros(max_it+1,max_it);
err = zeros(1,max_it);

% Arnoldi for (A,b)
V(:,1) = b/norm(b);
for j = 1:max_it
    w = A*V(:,j);
    mv = mv + 1;
    for reo = 0:reorth
        for i = 1:j
            h = V(:,i)'*w;
            H(i,j) = H(i,j) + h;
            w = w - V(:,i)*h;
        end
    end
    H(j+1,j) = norm(w);
    V(:,j+1) = w/H(j+1,j);

    % Every d iterations, compute either exact error or an estimate of the error
    % using a previous approximation
    if rem(j,d) == 0

        approx = V(:,1:j)*fm(H(1:j,1:j), norm(b)*eye(j,1));

        if err_monitor == "exact"

            err(d_it) = norm(exact - approx)/norm(exact);

        elseif err_monitor == "estimate"

            err(d_it) = norm(prev_approx - approx)/norm(b);
            prev_approx = approx;

        end

        if err(d_it) < tol
            break;
        end

        if j < max_it
            d_it = d_it + 1;
        end

    end
end

out.m = j;
out.approx = approx;
out.err = err(:,1:d_it);
out.mv = mv;
out.d_it = d_it;

end