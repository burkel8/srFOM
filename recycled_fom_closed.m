% FUNCTION: out = RECYCLED_FOM_CLOSED(A,b,param)
% A function which computes the recycled FOM approximation to f(A)b,
% using the closed form formula.

% INPUT:  A   The matrix
%         b   The vector
%        param An input struct with the following fields
%
%        param.max_it   The maximum number of Arnoldi iterations
%        param.n        The dimension of A
%        param.reorth   Re-orthogonalization parameter
%        param.fm       fm = @(A,b) f(A)*b
%        param.tol      Convergence tolerance
%        param.U        Basis for recycling subspace
%        param.k        Dimension of recycling subspace

% OUTPUT: out           An output struct with the following fields
%
%         out.m         Number of Arnoldi iterations executed
%         out.approx    Approximation to f(A)b
%         out.err       Vector storing exact relative errors at each
%                       iteration
%         out.U         Updated recycling subspace

function out = recycled_fom_closed(A,b,param)

max_it = param.max_it;
n = param.n;
fm = param.fm;
reorth = param.reorth;
exact = param.exact;
tol = param.tol;
U = param.U;
k = param.k;
mv = 0;

d = param.d;
d_it = 1;
prev_approx = b;
err_monitor = param.err_monitor;

V = zeros(n,max_it);
H = zeros(max_it+1,max_it);
err = zeros(1,max_it);
err_est = zeros(1,max_it);

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

        W = [ V(:,1:j), U ];

        % Gram-Schmidt for U against V
        for l = j+1:size(W,2)
            w = W(:,l);
            for reo = 0:reorth
                for i = 1:l-1
                    h = W(:,i)'*w;
                    w = w - W(:,i)*h;
                end
            end
            w = w/norm(w);
            W(:,l) = w;
        end

        if isempty(U)
            HH = H(1:j,1:j);
        else
            AU = A*W(:,j+1:end);
            em = zeros(j,1); em(j) = 1;
            HH = [ H(1:j,1:j) ;
                H(j+1,j)*(W(:,j+1:end)'*V(:,j+1))*em' ];
            HH = [ HH , W'*AU ];
        end

        approx = W*fm(HH, norm(b)*eye(size(W,2),1));

        if err_monitor == "exact"

            err(d_it) = norm(exact - approx)/norm(exact);

        elseif err_monitor == "estimate"

            err(d_it) = norm(prev_approx - approx)/norm(b);
            prev_approx = approx;

        end

        if err(d_it) < tol
            fprintf("\n Early convergence at iteration %d \n", j);
            break;
        end

        if j < max_it
            d_it = d_it + 1;
        end
    end
end

mv = mv + k;
% get updated Ritz vectors
[X,T] = schur(HH);
ritz = ordeig(T);
[~,ind] = sort(abs(ritz),'ascend');
select = false(length(ritz),1);
select(ind(1:min(k,j))) = 1;
[X,T] = ordschur(X,T,select);  % H = X*T*X'

if T(min(k,j)+1,min(k,j))~=0   % don't tear apart 2x2 diagonal blocks of Schur factor
    keep = min(k,j)+1;
else
    keep = min(k,j);
end

out.U = W*X(:,1:keep);
out.m = j;
out.approx = approx;
out.err = err(:,1:d_it);
out.err_est = err_est(:,1:d_it);
out.mv = mv;
out.d_it = d_it;

end