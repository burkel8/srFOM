%% test_quad_v_closed.m

% This file tests our new closed form version of recycled fom for
% f(A)b and compares it to the original quadrature based implementation
% proposed in [2].

% The sequence of vectors is evaluated using the following methods
% fom: The standard fom approximation
% rfom: The recycled fom presented in [1]
% srfom: The sketched and recycled fom presented in [1].
% rFOM (quad): The quadrature based recycling algorithm from [2]

% [1] L. Burke, S. Güttel. - Krylov Subspace Recycling With Randomized Sketching For Matrix Functions,
% arXiv :2308.02290 [math.NA] (2023)

% [2] [2] L. Burke, A. Frommer, G. Ramirez, K. M Soodhalter -
% Krylov Subspace Recycling For Matrix Functions, arXiv :2209.14163 [math.NA] (2022)

close all
addpath(genpath('../'));
mydefaults
rng('default')

% The maximum number of iterations of the methods
max_it = 100;

% Boolean variable to descide if the Arnoldi vectors in fom rfom should
% be re-orthogonalized (set to 1), or not (set to 0) (default is 0)
reorth = 0;

% The error tolerance used to define convergence.
tol = 1e-15;

% The error tolerance for the SVD decomposition in stabilized srFOM
svd_tol = 1e-14;

% The dimension of the recycling subspace used by rfom and srfom
k = 30;

% Arnoldi truncation parameter for the truncated Arnoldi method used in
% sfom and srfom. Each Arnoldi vector is orthogonalized against the previous
% t vectors.
t = 2;

% A matrix whos columns span the recycling subspace (default empty)
U = [];

% The number of f(A)b vectors in the sequence to evaluate
num_problems = 30;

% sketching parameter (number of rows of sketched matrix S)
s = 400;

% "strength" of matrix perturbation (default 0, special
% case when matrix remains fixed throughout the sequence )
pert = 1e-08;

% Monitor error every d iterations
d = 1;

% err_monitor set to "estimate" or "exact" to determine if the error should
% be estimated or computed exactly.
err_monitor = "exact";

% Vectors of length num_problems which will store the final exact error
% (only meaningful if the max number of iterations were performed)
fom_err = zeros(1,num_problems);
rfom_closed_err = zeros(1,num_problems);
rfom_quad1_err = zeros(1,num_problems);
rfom_quad2_err = zeros(1,num_problems);
srfom_err = zeros(1,num_problems);

if ~0   % generate data
    fprintf("\nGenerating a sequence of %d matrices, vectors and exact solutions \n", num_problems);
    fprintf("\n This may take a while! \n");
    load("../data/conf6_0-4x4-30.mat");
    A = Problem.A;
    n = size(A,1);
    A = A + 6.0777*speye(n);
    rng('default')
    B = randn(n,30);
    pert = 1e-8;
    for j = 1:num_problems
        AA{j} = A;
        E{j} = sqrtm(full(A))\B;
        A = A + pert*sprandn(A);
        fprintf("\n Problem %d data generated!", j);
    end
    save qcdsqrt-8 AA B E
    fprintf("\n Finished generating data!\n \n");
else     % or load it
    fprintf("\n Loading Data\n");
    load qcdsqrt-8.mat
end

% Construct subspace embedding matrix
hS = srft(n,s);
n = size(B,1);

% Construct function handle which inputs a matrix A, vector b and returns
% f(A)*b where f is the inverse square root function
fm = @(X,v) sqrtm(full(X))\v;

% Set up algorithm input struct
param.max_it = max_it;
param.reorth = reorth;
param.fm = fm;
param.tol = tol;
param.U = U;
param.SU = [];
param.SAU = [];
param.k = k;
param.t = t;
param.s = s;
param.d = d;
param.err_monitor = err_monitor;
param.pert = pert;
param.n = n;
param.hS = hS;

% input structs for fom, rfom, srfom
fom_param = param;
rfom_quad1_param = param;
rfom_quad2_param = param;
rfom_closed_param = param;
srfom_param = param;

% Set param properties unique to differnt methods
rfom_quad1_param.num_quad_points = 50;
rfom_quad2_param.num_quad_points = 100;
srfom_param.svd_tol = svd_tol;

% Loop through all systems
fprintf("\n #### Evaluating a sequence of %d f(A)b applications ####   \n", num_problems);

for i = 1:num_problems

    fprintf("\n #### Problem %d #### \n", i);

    A = AA{i};
    b = B(:,i);
    exact = E{i}(:,i);

    fom_param.exact = exact;
    rfom_closed_param.exact = exact;
    rfom_quad1_param.exact = exact;
    rfom_quad2_param.exact = exact;
    srfom_param.exact = exact;

    % Compute standard Arnoldi approximation.
    fprintf("\n Computing fom approximation .... \n");
    [fom_out] = fom(A,b,fom_param);
    fom_err(i) = fom_out.err(fom_out.m);

    % Compute the recycled FOM (quadrature based) approximation, assign the output recycling
    % subspace to be the input recycling subspace for the next problem.
    fprintf("\n Computing rfom (quad 50) approximation .... \n");
    rfom_quad1_out = recycled_fom_quad(A,b,rfom_quad1_param);
    rfom_quad1_param.U = rfom_quad1_out.U;
    rfom_quad1_err(i) = rfom_quad1_out.err;

    % Compute the recycled FOM (quadrature based) approximation, assign the output recycling
    % subspace to be the input recycling subspace for the next problem.
    fprintf("\n Computing rfom (quad 100) approximation .... \n");
    rfom_quad2_out = recycled_fom_quad(A,b,rfom_quad2_param);
    rfom_quad2_param.U = rfom_quad2_out.U;
    rfom_quad2_err(i) = rfom_quad2_out.err;

    % Compute the recycled FOM (closed) approximation, assign the output recycling
    % subspace to be the input recycling subspace for the next problem.
    fprintf("\n Computing rfom approximation .... \n");
    rfom_closed_out = recycled_fom_closed(A,b,rfom_closed_param);
    rfom_closed_param.U = rfom_closed_out.U;
    rfom_closed_err(i) = rfom_closed_out.err(rfom_closed_out.m);

    % Compute the sketched and recycled FOM approximation, assign the output
    % recycling subspace to be the input recycling subspace for the next
    % problem.

    % First call the method which uses sketched Rayleigh-Ritz
    fprintf("\n Computing srfom (with sRR) approximation .... \n");
    srfom_out = sketched_recycled_fom_stabilized(A,b,srfom_param);
    srfom_param.U = srfom_out.U;
    srfom_param.SU = srfom_out.SU;
    srfom_param.SAU = srfom_out.SAU;
    srfom_err(i) = srfom_out.err(srfom_out.m);
end

figure
semilogy(fom_err,'-');
grid on;
hold on
semilogy(rfom_quad1_err,'-o');
semilogy(rfom_quad2_err,'-*');
semilogy(rfom_closed_err,'V-');
semilogy(srfom_err,'+--');
legend('FOM','rFOM (quad) 50','rFOM (quad) 100','rFOM (Algorithm 2.1)','srFOM (Algorithm 3.1)');
xlabel('problem')
ylabel('relative error')




