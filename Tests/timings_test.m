% The definiton of the test inputs is given below

clear all
close all
rng('default')

% The maximum number of iterations of the methods
max_it = 400;

% Boolean variable to descide if the Arnoldi vectors in fom
% rfom should be re-orthogonalized (set to 1), or not (set to 0)
% (default is 0)
reorth = 0;

% The error tolerance used to define convergence.
tol = 1e-3;

% Error tolerance of SVD decomposition for stabilization
svd_tol = 1e-14;

% The dimension of the recycling subspace used by rfom and srfom
k = 30;

% Arnoldi truncation parameter for the truncated Arnoldi method
% used in sfom and srfom. Each Arnoldi vector is
% orthogonalized against the previous t vectors (default t = 2)
t = 2;

% A matrix whos columns span the recycling subspace (default empty)
U = [];

% The number of f(A)b vectors in the sequence to evaluate
num_problems = 30;

% sketching parameter (number of rows of sketched matrix S)
s = 600;

% represents "strength" of matrix perturbation (default 0, special
% case when matrix remains fixed throughout the sequence)
pert = 0;

% Compute exact error or estimate error every d iterations
d = 5;

% err_monitor set to "estimate" or "exact" to determine if the error should
% be estimated or computed exactly.
err_monitor = 'estimate';

addpath(genpath('../'));

% Generate Poisson Matrix
n = 10609;
A = gallery('neumann', n) + 0.0001*speye(n);

% Construct function handle which inputs a matrix A, vector b and returns
% f(A)*b where f is the inverse function
fm = @(X,v) full(X)\v;

% Set up algorithm input struct
param.max_it = max_it;
param.n = n;  % SG: don't really need this as param; use size(A,1)
param.reorth = reorth;
param.fm = fm;
param.tol = tol;
param.U = U;
param.k = k;
param.t = t;
param.svd_tol = svd_tol;
param.s = s;     % SG: could be avoided by sketching b and getting its length
param.d = d;
param.err_monitor = err_monitor;
param.pert = pert;

% input structs for fom, rfom, srfom
fom_param = param;
rfom_param = param;
sfom_param = param;
srfom_sRR_param = param;
srfom_stabsRR_param = param;
srfom_stabsRR_optim_param = param;


rng(1)
B = rand(n,num_problems);

fprintf("\n ### FOM ### \n");
tic
for i = 1:num_problems
    b = B(:,i);
    [fom_out] = fom(A,b,fom_param);
    err_fom(i) = norm(fom_out.approx - A\b)/norm(b);
end
toc

fprintf("\n ### sFOM ### \n");
tic
% Construct subspace embedding matrix
hS = srft(n,s);
for i = 1:num_problems
    sfom_param.hS = hS;
    b = B(:,i);
   [sfom_out] = sketched_fom(A,b,sfom_param);
   err_sfom(i) = norm(sfom_out.approx - A\b)/norm(b);
end
toc

fprintf("\n ### rFOM ### \n");
tic
for i = 1:num_problems
    b = B(:,i);
    [rfom_out] = recycled_fom_closed(A,b,rfom_param);
    err_rfom(i) = norm(rfom_out.approx - A\b)/norm(b);
    rfom_param.U = rfom_out.U;
end
toc

fprintf("\n ### srFOM  ### \n");
rng('default')
tic
% Construct subspace embedding matrix
hS = srft(n,s);
srfom_sRR_param.hS = hS;
srfom_sRR_param.U = [];
srfom_sRR_param.SU = [];  % SG: reset SU and SAU as well
srfom_sRR_param.SAU = [];
for i=1:num_problems
   
    b = B(:,i);
    [srfom_sRR_out] = sketched_recycled_fom(A,b,srfom_sRR_param);
    err_srfom_sRR(i) = norm(srfom_sRR_out.approx - A\b)/norm(b);
    srfom_sRR_param.U = srfom_sRR_out.U;
    srfom_sRR_param.SU = srfom_sRR_out.SU;   % SG: also provide sketched U
    srfom_sRR_param.SAU = srfom_sRR_out.SAU; 
end
toc

%%
fprintf("\n ### srFOM (stab) ### \n");
rng('default')
tic
% Construct subspace embedding matrix
hS = srft(n,s);
srfom_stabsRR_optim_param.hS = hS;
srfom_stabsRR_optim_param.U = [];
srfom_stabsRR_optim_param.SU = [];  % SG: reset SU and SAU as well
srfom_stabsRR_optim_param.SAU = [];
for i=1:num_problems
     b = B(:,i);
    [srfom_stabsRR_optim_out] = sketched_recycled_fom_stabilized(A,b,srfom_stabsRR_optim_param);
    err_srfom_stabsRR_optim(i) = norm(srfom_stabsRR_optim_out.approx - A\b)/norm(b);
    srfom_stabsRR_optim_param.U = srfom_stabsRR_optim_out.U;
    srfom_stabsRR_optim_param.SU = srfom_stabsRR_optim_out.SU;   % SG: also provide sketched U
    srfom_stabsRR_optim_param.SAU = srfom_stabsRR_optim_out.SAU; %     and sketched A*U
end
toc

