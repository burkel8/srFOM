% FUNCTION: out = SKETCHED_RECYCLED_FOM(A,b,param)
% A function which computes the sketched and recycled FOM approximation to f(A)b

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
%        param.t        Truncated Arnoldi truncation parameter
%        param.hS       Subspace embedding matrix
%        param.s        Number of rows of subspace embedding matrix

% OUTPUT: out           An output struct with the following fields
%
%         out.m         Number of Arnoldi iterations executed
%         out.approx    Approximation to f(A)b
%         out.err       Vector storing exact relative errors at each
%                       iteration
%         out.U         Updated recycling subspace

function [out] = sketched_recycled_fom(A,b,param)

if isnumeric(A)
    A = @(v) A*v;
end

max_it = param.max_it;
n = param.n;
fm = param.fm;
tol = param.tol;
hS = param.hS;
t = param.t;
U = param.U;
k = param.k;
err_monitor = param.err_monitor;
d = param.d;
s = param.s;
mv = 0;
ip = 0;
sv = 0;
d_it = 1;
prev_approx = b;

V = zeros(n,max_it+1);
SV = zeros(s,max_it+1);
H = zeros(max_it+1,max_it);
err = zeros(1,max_it);

if isempty(U)
    SAW = [];
    SW = [];
else

    % In the special case when the matrix does not change, we can re-use SU
    % from previous problem,
    if param.mat_change == 0
        SW = param.SU;
        SAW = param.SAU;
        mv = mv + 0;
    else
        SW = param.SU;
        SAW = hS(A(U));
        mv = mv + size(U,2);
        sv = sv + size(U,2);
    end
end

% Arnoldi for (A,b)
Sb = hS(b);              % SG: moved this out of for-j loop as only done once
sv = sv + 1;
SV(:,1) = Sb/norm(b);
V(:,1) = b/norm(b);
for j = 1:max_it
    w = A(V(:,j));
    mv = mv + 1;

    for i = max(j-t+1,1):j
        H(i,j) = V(:,i)'*w;
        w = w - V(:,i)*H(i,j);
        ip = ip + 1;
    end

    H(j+1,j) = norm(w);
    ip = ip + 1;
    V(:,j+1) = w/H(j+1,j);
    SV(:,j+1) = hS(V(:,j+1));
    sv = sv + 1;

    % Every d iterations, compute either exact error or an estimate of the error
    % using a previous approximation
    if rem(j,d) == 0 || j == max_it

        SW = [ param.SU, SV(:,1:j) ];

        % No need to sketch A*V since S*A*V = (S*V)*H
        SAV = SV(:,1:j+1)*H(1:j+1,1:j);
        SAW = [ param.SAU, SAV ];

        [Q,R] = qr(SW,0);
        coeffs = R\fm(Q'*SAW/R, Q'*Sb);

        % Update approximation without explicitly forming [U V(:,1:j)]
        if size(U,2) > 0
            approx = U*coeffs(1:size(U,2),1) + V(:,1:j)*coeffs(size(U,2)+1:end,1);
        else
            approx = V(:,1:j)*coeffs(size(U,2)+1:end,1);
        end

        % Compute either exact error or an estimate baseed off a previous
        % approximation
        if err_monitor == "exact"
            exact = param.exact;
            err(d_it) = norm(exact - approx)/norm(exact);
        elseif err_monitor == "estimate"
            err(d_it) = norm(prev_approx - approx)/norm(b);
            prev_approx = approx;
        end

        % Early convergence of srFOM
        if err(d_it) < tol
            break
        end

        if j < max_it
            d_it = d_it + 1;
        end

    end
end

% get updated Ritz vectors using sketched Rayleigh-Ritz
H = R\(Q'*SAW);  % = pinv(SW)*SAW
[X,T] = schur(H);
ritz = ordeig(T);
[~,ind] = sort(abs(ritz),'ascend');
select = false(length(ritz),1);
select(ind(1:k)) = 1;
[X,T] = ordschur(X,T,select);  % H = X*T*X'
if T(k+1,k)~=0   % don't tear apart 2x2 diagonal blocks of Schur factor
    keep = k+1;
else
    keep = k;
end

% cheaper update of recycling subspace by avoiding explicilty constructing the
% matrix [U V(:,1:j)]
if size(U,2) > 0
    out.U = U*X(1:size(U,2),1:keep) + V(:,1:j)*X(size(U,2)+1:end,1:keep);
else
    out.U = V(:,1:j)*X(size(U,2)+1:end,1:keep);
end

out.SU = SW*X(:,1:keep);
out.SAU = SAW*X(:,1:keep);
out.k = keep;
out.m = j;
out.approx = approx;
out.err = err(:,1:d_it);
out.mv = mv;
out.ip = ip;
out.sv = sv;
out.d_it = d_it;

end