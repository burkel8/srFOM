%% test_quad_v_closed.m

% This file tests our new closed form version of recycled fom for
% f(A)b by comparing it to the original quadrature based implementation
% proposed in [Liam, Kirk, Andreas, Gustavo] paper.

% We test our method on the inverse square root function using on a fixed QCD
% matrix with changing right hand sides.

% The sequence of vectors is evaluated using the following 4 methods
% fom: The standard Arnoldi approximation
% rfom: The recycled fom presented in [Stefan and Liam work]
% sfom: The sketched fom presented in [Stefan and Marcel]
% srfom: The sketched and recycled fom presented in [Stefan and Liam work].

% The definiton of the test inputs is given below

clear all
close all
rng('default')

% The maximum number of iterations of the methods
max_it = 100;

% Boolean variable to descide if the Arnoldi vectors in fom rfom should
% be re-orthogonalized (set to 1), or not (set to 0) (default is 0)
reorth = 0;

% The error tolerance used to define convergence.
tol = 1e-15;

% The error tolerance for the SVD decomposition in stabilized srFOM
svd_tol = 1e-15;

% The dimension of the recycling subspace used by rfom and srfom
k = 30;

% Arnoldi truncation parameter for the truncated Arnoldi method used in
% sfom and srfom. Each Arnoldi vector is orthogonalized against the previous
% t vectors (default t = 3)
t = 3;

% A matrix whos columns span the recycling subspace (default empty)
U = [];

% The number of f(A)b vectors in the sequence to evaluate
num_problems = 20;

% sketching parameter (number of rows of sketched matrix S)
s = 400;

% "strength" of matrix perturbation (default 0, special
% case when matrix remains fixed throughout the sequence )
pert = 0.00000001;

d = 1;
err_monitor = "exact";

addpath(genpath('../'));

% Load QCD matrix
load("../data/conf6_0-4x4-30.mat");
A = Problem.A;
n = size(A,1);
A = A + 6.0777*speye(n);

% Precompute square root of A
load("../data/SqA");

% Construct subspace embedding matrix
hS = srft(n,s);

% Construct function handle which inputs a matrix A, vector b and returns
% f(A)*b where f is the inverse square root function
fm = @(X,v) sqrtm(full(X))\v;

% Set up algorithm input struct
param.max_it = max_it;
param.n = n;
param.reorth = reorth;
param.fm = fm;
param.tol = tol;
param.U = U;
param.SU = [];
param.SAU = [];
param.k = k;
param.t = t;
param.hS = hS;
param.s = s;
param.d = d;
param.err_monitor = err_monitor;

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

% Vectors of length num_problems which will store the final exact error
% (only meaningful if the max number of iterations were performed)
fom_err = zeros(1,num_problems);
rfom_closed_err = zeros(1,num_problems);
rfom_quad1_err = zeros(1,num_problems);
rfom_quad2_err = zeros(1,num_problems);
srfom_err = zeros(1,num_problems);

% Loop through all systems
fprintf("\n #### Evaluating a sequence of %d f(A)b applications ####   \n", num_problems);
for i = 1:num_problems

    fprintf("\n #### Problem %d #### \n", i);

    % Create b vector
    rng(i);
    b = randn(n,1);

    % Compute exact solution and assign to the input struct for each method.
    % If the matrix remains fixed for each problem in the sequence, use
    % precomputed square root of A, or else we must compute the exact solution
    % from scratch.

    % We may also use the precomputed square root for the first problem in the
    % sequence.
    if i == 1
        exact = SqA\b;
    else
        if pert == 0
            exact = SqA\b;
        else
            exact = fm(A,b);
        end
    end

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

    % Slowly change the matrix for next problem.
    A = A + pert*sprand(A);

end

figure
semilogy(fom_err,'-','LineWidth',2);
grid on;
hold on
semilogy(rfom_quad1_err,'-o','LineWidth',2);
semilogy(rfom_quad2_err,'-*','LineWidth',2);
semilogy(rfom_closed_err,'V-','LineWidth',2);
semilogy(srfom_err,'V--', 'LineWidth',2);
legend('FOM','rFOM (quad) 50','rFOM (quad) 100','rFOM (Alg.2.1)','srFOM (Alg. 3.1)','interpreter','latex','FontSize',13);
xlabel('problem','FontSize',13)
ylabel('relative error','FontSize',13)
mypdf('fig/compare_quad_v_closed',.66,1.4)
hold off;
pause(3)




