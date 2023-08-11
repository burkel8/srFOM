% FUNCTION: out = RECYCLED_FOM_CLOSED_LARGEST(A,b,param)
% A function which computes the recycled FOM approximation to f(A)b,
% using the closed form formula.

% INPUT:  A   The matrix or function handle
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

function out = recycled_fom_closed_largest(A,b,param)

if isnumeric(A)
    A = @(v) A*v;
end

max_it = param.max_it;
n = param.n;
fm = param.fm;
reorth = param.reorth;

tol = param.tol;
U = param.U;
mv = 0;
ip = 0;
if isempty(U)
    AU = [];
else
    if param.pert == 0
        AU = param.AU;
    else
        AU = A(U); 
        mv = mv + size(U,2);
    end
end
k = param.k;
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
    w = A(V(:,j));
    mv = mv + 1;
    for reo = 0:reorth
        for i = 1:j
            h = V(:,i)'*w;
            ip = ip + 1;
            H(i,j) = H(i,j) + h;
            w = w - V(:,i)*h;
        end
    end
    H(j+1,j) = norm(w);
    ip = ip + 1;
    V(:,j+1) = w/H(j+1,j);

    % Every d iterations, compute either exact error or an estimate of the error
    % using a previous approximation

    if rem(j,d) == 0 || j == max_it

        % If we do not yet have U, then the approximation is equivelant to
        % the standard Arnoldi approximation
        if isempty(U)
            HH = H(1:j,1:j);
            coeffs = fm(HH, norm(b)*eye(j,1));
            approx = V(:,1:j)*coeffs(1:j,1);

            
        else  % Or else if we do have a U...

            C = V(:,1:j)'*U;
            U = U - V(:,1:j)*C;
            AU = AU - V(:,1:j+1)*(H(1:j+1,1:j)*C);   % AU = AU - A*V(:,1:j)*C
            [U,R] = qr(U,'econ');
            AU = AU/R;

            em = zeros(j,1); em(j) = 1;

            HH = [ H(1:j,1:j)  V(:,1:j)'*AU;
                H(j+1,j)*(U'*V(:,j+1))*em' U'*AU];

            ip = ip + 2*j*size(U,2) + size(U,2)^2; % for computing V'*U, V'*AU, U'*AU

            % Update approximation without explicitly forming [V(:,1:j) U]
            coeffs = fm(HH, norm(b)*eye(size(HH,2),1));
            approx =  V(:,1:j)*coeffs(1:j,:) + U*coeffs(j+1:end,:);

        end

        % Compute either an exact relative error, or else an estimate or
        % the error from a previous iteration
        if err_monitor == "exact"

            exact = param.exact;
            err(d_it) = norm(exact - approx)/norm(exact);

        elseif err_monitor == "estimate"

            err(d_it) = norm(prev_approx - approx)/norm(b);
            prev_approx = approx;

        end

        % Early convergence of rFOM
        if err(d_it) < tol
            break
        end

        if j < max_it
            d_it = d_it + 1;
        end
    end
end

% get updated Ritz vectors
[X,T] = schur(HH);
ritz = ordeig(T);
[~,ind] = sort(real(ritz),'descend');
select = false(length(ritz),1);
select(ind(1:min(k,j))) = 1;
[X,T] = ordschur(X,T,select);  % H = X*T*X'

if T(min(k,j)+1,min(k,j))~=0   % don't tear apart 2x2 diagonal blocks of Schur factor
    keep = min(k,j)+1;
else
    keep = min(k,j);
end

% cheaper update of recycling subspace by avoiding explictly constructing 
% the matrix [V(:,1:j) U]
if size(U,2) > 0
    out.U = V(:,1:j)*X(1:j,1:keep) + U*X(j+1:end,1:keep);
    out.AU = V(:,1:j+1)*(H(1:j+1,1:j)*X(1:j,1:keep)) + AU*X(j+1:end,1:keep);
else
    out.U = V(:,1:j)*X(:,1:keep);
    out.AU = V(:,1:j+1)*(H(1:j+1,1:j)*X(1:j,1:keep));
end

out.m = j;
out.approx = approx;
out.err = err(:,1:d_it);
out.err_est = err_est(:,1:d_it);
out.mv = mv;
out.ip = ip;
out.d_it = d_it;

end