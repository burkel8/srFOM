function [out] = sketched_recycled_fom_stabilized(A,b,param)
% FUNCTION: out = SKETCHED_RECYCLED_FOM_STABILIZED(A,b,param)
% A function which computes the sketched and recycled FOM (with
% stabilization) approximation to f(A)b
%
% INPUT:  A       n-by-n matrix
%         b       n-by-1 vector
%         param   struct with the following fields:
%
%         param.max_it   The maximum number of Arnoldi iterations
%         param.n        The dimension of A
%         param.reorth   Re-orthogonalization parameter
%         param.fm       fm = @(A,b) f(A)*b
%         param.tol      Convergence tolerance
%         param.U        Basis for recycling subspace
%         param.k        Dimension of recycling subspace
%         param.t        Truncated Arnoldi truncation parameter
%         param.hS       Subspace embedding matrix
%         param.s        Number of rows of subspace embedding matrix
%
% OUTPUT: out            struct with the following fields
%
%         out.m          Number of Arnoldi iterations executed
%         out.approx     Approximation to f(A)b
%         out.err        Vector storing exact relative errors at each
%                        iteration
%         out.U          Updated recycling subspace


max_it = param.max_it;
n = param.n;
fm = param.fm;
exact = param.exact;
tol = param.tol;
hS = param.hS;
t = param.t;
U = param.U;
k = param.k;
d = param.d;
svd_tol = param.svd_tol;

mv = 0;
d_it = 1;
prev_approx = b;
err_monitor = param.err_monitor;

V = zeros(n,max_it+1);
H = zeros(max_it+1,max_it);
err = zeros(1,max_it);

if isempty(U)
    SAW = [];
    SW = [];
else
    SW = hS(U);
    AU = A*U;
    mv = mv + k;
    SAW = hS(AU);
end

% Arnoldi for (A,b)
V(:,1) = b/norm(b);
for j = 1:max_it
    w = A*V(:,j);
    mv = mv + 1;
    SAW = [SAW, hS(w)];

    for i = max(j-t+1,1):j
        H(i,j) = V(:,i)'*w;
        w = w - V(:,i)*H(i,j);
    end
    H(j+1,j) = norm(w);
    V(:,j+1) = w/H(j+1,j);

    W = [ U, V(:,1:j) ];

    % augmented sketched Arnoldi approx
    SW = [SW hS(V(:,j))];
    Sb = hS(b);

    % Every d iterations, compute either exact error or an estimate of the error
    % using a previous approximation
    if rem(j,d) == 0

        % Compute economic SVD of SW
        [Lfull,Sigfull,Jfull] = svd(SW,'econ');

        % Truncate SVD
        ell = find(diag(Sigfull) > svd_tol, 1, 'last');
        L = Lfull(:,1:ell);
        Sig = Sigfull(1:ell,1:ell);
        J = Jfull(:,1:ell);
        H = L'*SAW*J;
        approx = W*(J*(Sig\fm(H/Sig,L'*Sb)));

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

%% update augmentation space using QZ
if isreal(H) && isreal(Sig)
    [AA, BB, Q, Z] = qz(H,Sig,'real'); % Q*A*Z = AA, Q*B*Z = BB
else
    [AA, BB, Q, Z] = qz(H,Sig);
end
ritz = ordeig(AA,BB);
[~,ind] = sort(abs(ritz),'ascend');
select = false(length(ritz),1);
select(ind(1:k)) = 1;
[AA,BB,Q,Z] = ordqz(AA,BB,Q,Z,select);
if AA(k+1,k)~=0 || BB(k+1,k)~=0  % don't tear apart 2x2 diagonal blocks
    keep = k+1;
else
    keep = k;
end

out.U = W*(J*Z(:,1:keep));
out.m = j;
out.mv = mv;
out.approx = approx;
out.err = err(:,1:d_it);
out.d_it = d_it;

end