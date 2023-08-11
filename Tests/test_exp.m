%% test_exp.m

% This file evaluates a sequence of vectors of the form f(A)b where f is
% the exponential function

% The sequence of vectors is evaluated using the following 4 methods
% fom: The standard Arnoldi approximation
% rfom: The recycled fom presented in [Stefan and Liam work]
% sfom: The sketched fom presented in [Stefan and Marcel]
% srfom: The sketched and recycled fom presented in [Stefan and Liam work].

% The definiton of the test inputs is given below

clear all
close all
mydefaults
rng('default')

% The maximum number of iterations of the methods
max_it = 400;

% Boolean variable to descide if the Arnoldi vectors in fom
% rfom should be re-orthogonalized (set to 1), or not (set to 0)
% (default is 0)
reorth = 0;

% The error tolerance used to define convergence.
tol = 1e-9;

% Error tolerance of SVD decompositon
svd_tol = 1e-14;

% The dimension of the recycling subspace used by rfom and srfom
k = 50;

% Arnoldi truncation parameter for the truncated Arnoldi method
% used in sfom and srfom. Each Arnoldi vector is
% orthogonalized against the previous t vectors (default t = 3)
t = 2;

% A matrix whos columns span the recycling subspace (default empty)
U = [];

% The number of f(A)b vectors in the sequence to evaluate
num_problems = 30;

% sketching parameter (number of rows of sketched matrix S)
s = 400;

% represents "strength" of matrix perturbation (default 0, special
% case when matrix remains fixed throughout the sequence )
pert = 0;

% Compute exact error or estimate error every d iterations
d = 10;

% exponential time step
tt = 0.01;

% err_monitor set to "estimate" or "exact" to determine if the error should
% be estimated or computed exactly.
err_monitor = "estimate";

addpath(genpath('../'));

load("data/cdBeispiel.mat"); A = sparse(A);
n = size(A,1);
rng(1);
b = rand(n,1);

% Construct subspace embedding matrix
hS = srft(n,s);

% Construct function handle which inputs a matrix A, vector b and returns
% f(A)*b where f is the exponential function
fm = @(X,v) expm(tt*full(X))*v;
ExA = expm(tt*full(A));

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
param.pert = pert;

% input structs for fom, rfom, srfom
fom_param = param;
rfom_param = param;
sfom_param = param;
srfom_sRR_param = param;
srfom_stabsRR_param = param;

srfom_stabsRR_param.svd_tol = 1e-14;

% Vectors of length num_problems which will store the total Arnoldi cycle
% length needed for each problem in the sequence to converge.
fom_m = zeros(1,num_problems);
rfom_m = zeros(1,num_problems);
sfom_m = zeros(1,num_problems);
srfom_sRR_m = zeros(1,num_problems);
srfom_stabsRR_m = zeros(1,num_problems);

% Vectors of length num_problems which will store the total number
% of MATVEC's needed for each problem in the sequence to converge.
fom_mv = zeros(1,num_problems);
rfom_mv = zeros(1,num_problems);
sfom_mv = zeros(1,num_problems);
srfom_sRR_mv = zeros(1,num_problems);
srfom_stabsRR_mv = zeros(1,num_problems);

% Vectors of length num_problems which will store the final exact error
% (only meaningful if the max number of iterations were performed)
fom_err = zeros(1,num_problems);
rfom_err = zeros(1,num_problems);
sfom_err = zeros(1,num_problems);
srfom_sRR_err = zeros(1,num_problems);
srfom_stabsRR_err = zeros(1,num_problems);

% Loop through all systems
fprintf("\n #### Evaluating a sequence of %d f(A)b applications ####   \n", num_problems);
for i = 1:num_problems

    fprintf("\n #### Problem %d #### \n", i);

    %b = randn(n,1);
    exact = ExA*b;

    fom_param.exact = exact;
    rfom_param.exact = exact;
    sfom_param.exact = exact;
    srfom_sRR_param.exact = exact;
    srfom_stabsRR_param.exact = exact;

    % Compute standard Arnoldi approximation.
    fprintf("\n Computing fom approximation .... \n");
    [fom_out] = fom(A,b,fom_param);
    fom_approx = fom_out.approx;
    fom_m(i) = fom_out.m;
    fom_mv(i) = fom_out.mv;
    fom_err(i) = norm(fom_out.approx - exact)/norm(b);
    
    % Compute the recycled FOM approximation, assign the output recycling
    % subspace to be the input recycling subspace for the next problem.
    fprintf("\n Computing rfom approximation .... \n");
    [rfom_out] = recycled_fom_closed(A,b,rfom_param);
    rfom_param.U = rfom_out.U;
    rfom_param.AU = rfom_out.AU;
    rfom_m(i) = rfom_out.m;
    rfom_mv(i) = rfom_out.mv;
    rfom_err(i) = norm(rfom_out.approx - exact)/norm(b);

    % Compute sketched FOM approximation
    fprintf("\n Computing sfom approximation .... \n");
    sfom_out = sketched_fom(A,b,sfom_param);
    sfom_m(i) = sfom_out.m;
    sfom_mv(i) = sfom_out.mv;
    sfom_err(i) = norm(sfom_out.approx - exact)/norm(b);

    % Compute the sketched and recycled FOM approximation, assign the output
    % recycling subspace to be the input recycling subspace for the next
    % problem.

    % First call the method which uses sketched Rayleigh-Ritz
    fprintf("\n Computing srfom (with sRR) approximation .... \n");
    srfom_sRR_out = sketched_recycled_fom(A,b,srfom_sRR_param);
    srfom_sRR_param.U = srfom_sRR_out.U;
    srfom_sRR_param.SU = srfom_sRR_out.SU;
    srfom_sRR_param.SAU = srfom_sRR_out.SAU;
    srfom_sRR_m(i) = srfom_sRR_out.m;
    srfom_sRR_mv(i) = srfom_sRR_out.mv;
    srfom_sRR_err(i) = norm(srfom_sRR_out.approx - exact)/norm(b);

    % Then, call the method which uses the stabilized sketched Rayleigh-Ritz
    fprintf("\n Computing srfom (with stabilized sRR) approximation .... \n");
    srfom_stabsRR_out = sketched_recycled_fom_stabilized(A,b,srfom_stabsRR_param);
    srfom_stabsRR_param.U = srfom_stabsRR_out.U;
    srfom_stabsRR_param.SU = srfom_stabsRR_out.SU;
    srfom_stabsRR_param.SAU = srfom_stabsRR_out.SAU;
    srfom_stabsRR_m(i) = srfom_stabsRR_out.m;
    srfom_stabsRR_mv(i) = srfom_stabsRR_out.mv;
    srfom_stabsRR_err(i) = norm(srfom_stabsRR_out.approx - exact)/norm(b);

    % Slowly change the matrix for next problem.
    %A = A + pert*sprand(A);

    % Take new b to be FOM approximation.
    b = fom_approx;
    
end


%% Plot final error of each problem in the sequence
figure
semilogy(fom_err,'-');
grid on;
hold on
semilogy(sfom_err,'--');
semilogy(rfom_err,'V-');
semilogy(srfom_sRR_err,'+--');
semilogy(srfom_stabsRR_err,'s--');
semilogy([0,30],[1e-9,1e-9],'k:')
legend('FOM','sFOM','rFOM','srFOM','srFOM (stab)','target tol','Location','southwest');
xlabel('problem')
ylabel('relative error')
title('exponential, actual error')
ylim([1e-16,1e-8])
mypdf('fig/exp_error_curves',.66*1.2,1.4/1.2)
hold on;
shg

%%
%Plot the Arnoldi cycle length needed for each problem to converge.
figure
plot(fom_m,'-');
grid on;
hold on;
plot(sfom_m,'--');
plot(rfom_m,'V-');
semilogy(srfom_sRR_m,'+--');
semilogy(srfom_stabsRR_m,'s--');
legend('FOM','sFOM', 'rFOM','srFOM','srFOM (stab)');
xlabel('problem')
ylabel('m');
title('exponential, adaptive m via estimator')
ylim([10,130])
mypdf('fig/exp_adaptive',.66*1.2,1.4/1.2)
shg


