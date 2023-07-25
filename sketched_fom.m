% FUNCTION: out = SKETCHED_FOM(A,b,param)
% A function which computes the sketched Arnoldi approximation to f(A)b

% INPUT:  A   The matrix
%         b   The vector
%        param An input struct with the following fields
%
%        param.max_it   The maximum number of Arnoldi iterations
%        param.n        The dimension of A
%        param.reorth   Re-orthogonalization parameter
%        param.fm       fm = @(A,b) f(A)*b
%        param.tol      Convergence tolerance
%        param.t        Truncated Arnoldi truncation parameter
%        param.hS       Subspace embedding matrix
%        param.s        Number of rows of subspace embedding matrix

% OUTPUT: out           An output struct with the following fields
%
%         out.m         Number of Arnoldi iterations executed
%         out.approx    Approximation to f(A)b
%         out.err       Vector storing exact relative errors at each
%                       iteration

function [out] = sketched_fom(A,b,param)

max_it = param.max_it;
n = param.n;
fm = param.fm;

tol = param.tol;
hS = param.hS;
t = param.t;
s = param.s;
d = param.d;
err_monitor = param.err_monitor;
mv = 0;

d_it = 1;
prev_approx = b;

V = zeros(n,max_it+1);
H = zeros(max_it+1,max_it);
SW = zeros(s,max_it+1);
err = zeros(1,max_it);

Sb = hS(b);
SW(:,1) = Sb/norm(b);
V(:,1) = b/norm(b);

for j = 1:max_it
    w = A*V(:,j);
    mv = mv + 1;

    for i = max(j-t+1,1):j
        H(i,j) = V(:,i)'*w;
        w = w - V(:,i)*H(i,j);
    end

    H(j+1,j) = norm(w);
    V(:,j+1) = w/H(j+1,j);
    SW(:,j+1) = hS(V(:,j+1));

    % Every d iterations, compute either exact error or an estimate of the error
    % using a previous approximation
    if rem(j,d) == 0

        SAW = SW(:,1:j+1)*H(1:j+1,1:j);

        [Q,R] = qr(SW(:,1:j),0);
        approx = V(:,1:j)*(R\fm(Q'*SAW(:,1:j)/R, Q'*Sb));

        if err_monitor == "exact"

            exact = param.exact;
            err(d_it) = norm(exact - approx)/norm(exact);

        elseif err_monitor == "estimate"

            err(d_it) = norm(prev_approx - approx)/norm(b);
            prev_approx = approx;

        end

        % Early convergence of sFOM
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