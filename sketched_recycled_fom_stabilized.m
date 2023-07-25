function [out] = sketched_recycled_fom_stabilized(A,b,param)
% FUNCTION: out = SKETCHED_RECYCLED_FOM_STABILIZED(A,b,param)
% A function which computes the sketched and recycled FOM (with
% stabilization) approximation to f(A)b
%
% INPUT:  A       n-by-n matrix
%         b       n-by-1 vector
%         param   struct with the following fields:
%
%         param.max_it   maximum number of Arnoldi iterations
%         param.reorth   re-orthogonalization parameter
%         param.fm       fm = @(A,b) f(A)*b
%         param.tol      convergence tolerance
%         param.U        basis for recycling subspace
%         param.SU, SAU  sketched recycling subspace
%         param.k        dimension of recycling subspace
%         param.t        Arnoldi truncation parameter
%         param.hS       subspace embedding matrix
%         param.s        number of rows of subspace embedding matrix
%         param.svd_tol  stabilization tolerance
%
% OUTPUT: out            struct with the following fields
%
%         out.m          number of Arnoldi iterations executed
%         out.approx     approximation to f(A)b
%         out.err        vector storing exact relative errors at each
%                        iteration
%         out.U          updated recycling subspace
%         out.SU, SAU    sketches of updated recycling subspace
%         out.mv         number of matrix-vector products


max_it = param.max_it;
n = param.n;
fm = param.fm;
tol = param.tol;
hS = param.hS;
t = param.t;
U = param.U;
k = param.k;
d = param.d;
s = param.s;
svd_tol = param.svd_tol;

mv = 0;
d_it = 1;
prev_approx = b;
err_monitor = param.err_monitor;

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
    if param.pert == 0
        SW = param.SU;
        SAW = param.SAU;
        mv = mv + 0;
    else
        SW = param.SU;
        SAW = hS(A*U);
        mv = mv + k;
    end
end

% Arnoldi for (A,b)
Sb = hS(b);
SV(:,1) = Sb/norm(b);
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
    SV(:,j+1) = hS(V(:,j+1));

    % Every d iterations, compute either exact error or an estimate
    % of the error using a previous approximation
    if rem(j,d) == 0

        SW = [ param.SU, SV(:,1:j) ];

        % No need to sketch A*V since S*A*V = (S*V)*H
        SAV = SV(:,1:j+1)*H(1:j+1,1:j);
        SAW = [ param.SAU, SAV ];

        % Compute economic SVD of SW
        [Lfull,Sigfull,Jfull] = svd(SW,'econ');

        % Truncate SVD
        ell = find(diag(Sigfull) > svd_tol, 1, 'last');
        L = Lfull(:,1:ell);
        Sig = Sigfull(1:ell,1:ell);
        J = Jfull(:,1:ell);
        HH = L'*SAW*J;

        % Update approximation without explicitly forming [U V(:,1:j)]
        coeffs = J*(Sig\fm(HH/Sig,L'*Sb));
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

% update augmentation space using QZ
if isreal(HH) && isreal(Sig)
    [AA, BB, Q, Z] = qz(HH,Sig,'real'); % Q*A*Z = AA, Q*B*Z = BB
else
    [AA, BB, Q, Z] = qz(HH,Sig);
end
ritz = ordeig(AA,BB);
[~,ind] = sort(abs(ritz),'ascend');
select = false(length(ritz),1);
select(ind(1:k)) = 1;
[AA,BB,~,Z] = ordqz(AA,BB,Q,Z,select);
if AA(k+1,k)~=0 || BB(k+1,k)~=0  % don't tear apart 2x2 diagonal blocks
    keep = k+1;
else
    keep = k;
end

% cheap update of recycling subspace without explicitly constructing [U V(:1:j)]
JZ = J*Z(:,1:keep);
if size(U,2) > 0
    out.U = U*JZ(1:size(U,2),:) + V(:,1:j)*JZ(size(U,2)+1:end,:);
else
    out.U = V(:,1:j)*JZ(size(U,2)+1:end,:);
end

out.SU = SW*JZ;
out.SAU = SAW*JZ;

out.m = j;
out.mv = mv;
out.approx = approx;
out.err = err(:,1:d_it);
out.d_it = d_it;

end