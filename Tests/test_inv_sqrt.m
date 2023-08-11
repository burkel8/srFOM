%% test_inv_sqrt.m

% This file evaluates a sequence of vectors of the form f(A)b where f is
% the inverse square root function and the matrix A is a QCD matrix which
% is purturbed as the sequence progresses. The right hand sides are
% randomly generated. 

% The sequence of vectors is evaluated using the following 4 methods
% fom: The standard Arnoldi approximation
% rfom: The recycled fom presented in [Stefan and Liam work]
% sfom: The sketched fom presented in [Stefan and Marcel]
% srfom: The sketched and recycled fom presented in [Stefan and Liam work].

% The definiton of the test inputs is given below

clear all
close all
rng('default')


%% %%%%%%%%% User Inputs %%%%%%%%%%%%%%
max_it = 100; % The maximum number of iterations of the methods 

reorth = 0;   % Boolean variable to descide if the Arnoldi vectors in fom 
              % rfom should be re-orthogonalized (set to 1), or not (set to 0)
              % (default is 0)

tol = 10e-15; % The error tolerance used to define convergence.
svd_tol = 1e-15;

k = 30;       % The dimension of the recycling subspace used by rfom and srfom

t = 3;        % Arnoldi truncation parameter for the truncated Arnoldi method
              % used in sfom and srfom. Each Arnoldi vector is
              % orthogonalized against the previous t vectors (default t = 3)

U = [];       % A matrix whos columns span the recycling subspace (default empty)

num_problems = 20; % The number of f(A)b vectors in the sequence to evaluate

s = 400; % sketching parameter (number of rows of sketched matrix S)

pert = 0; % represents "strength" of matrix perturbation (default 0, special
         % case when matrix remains fixed throughout the sequence )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
param.k = k;
param.t = t;
param.hS = hS;
param.s = s;

% input structs for fom, rfom, srfom
fom_param = param;
rfom_param = param;
sfom_param = param;
srfom_sRR_param = param;
srfom_stabsRR_param = param;

srfom_stabsRR_param.svd_tol = svd_tol;

% Vectors of length num_problems which will store the total Arnoldi cycle 
% length needed for each problem in the sequence to converge.
fom_m = zeros(1,num_problems);
rfom_m = zeros(1,num_problems);
sfom_m = zeros(1,num_problems);
srfom_sRR_m = zeros(1,num_problems);
srfom_stabsRR_m = zeros(1,num_problems);

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
rfom_param.exact = exact;
sfom_param.exact = exact;
srfom_sRR_param.exact = exact;
srfom_stabsRR_param.exact = exact;

% Compute standard Arnoldi approximation.
fprintf("\n Computing fom approximation .... \n");
fom_out = fom(A,b,fom_param);
fom_m(i) = fom_out.m;
fom_err(i) = fom_out.err(fom_out.m);

% Compute the recycled FOM approximation, assign the output recycling
% subspace to be the input recycling subspace for the next problem.
fprintf("\n Computing rfom approximation .... \n");
rfom_out = recycled_fom_closed(A,b,rfom_param);
rfom_param.U = rfom_out.U;
rfom_m(i) = rfom_out.m;
rfom_err(i) = rfom_out.err(rfom_out.m);

% Compute sketched FOM approximation
fprintf("\n Computing sfom approximation .... \n");
sfom_out = sketched_fom(A,b,sfom_param);
sfom_m(i) = sfom_out.m;
sfom_err(i) = sfom_out.err(sfom_out.m);

% Compute the sketched and recycled FOM approximation, assign the output
% recycling subspace to be the input recycling subspace for the next
% problem.

% First call the method which uses sketched Rayleigh-Ritz
fprintf("\n Computing srfom (with sRR) approximation .... \n");
srfom_sRR_out = sketched_recycled_fom(A,b,srfom_sRR_param);
srfom_sRR_param.U = srfom_sRR_out.U;
srfom_sRR_m(i) = srfom_sRR_out.m;
srfom_sRR_err(i) = srfom_sRR_out.err(srfom_sRR_out.m);

% Then, call the method which uses the stabilized sketched Rayleigh-Ritz
fprintf("\n Computing srfom (with stabilized sRR) approximation .... \n");
srfom_stabsRR_out = sketched_recycled_fom_stabilized(A,b,srfom_stabsRR_param);
srfom_stabsRR_param.U = srfom_stabsRR_out.U;
srfom_stabsRR_m(i) = srfom_stabsRR_out.m;
srfom_stabsRR_err(i) = srfom_stabsRR_out.err(srfom_stabsRR_out.m);

% Slowly change the matrix for next problem.
A = A + pert*sprand(A);

end

%%
% Plot final error of each problem in the sequence
figure
semilogy(fom_err,'-','LineWidth',2);
grid on;
hold on
semilogy(sfom_err,'--','LineWidth',2);
semilogy(rfom_err,'V-','LineWidth',2);
semilogy(srfom_sRR_err,'V--','LineWidth',2);
semilogy(srfom_stabsRR_err,'s--', 'LineWidth',2);
legend('FOM','sFOM','rFOM (Alg. 2.1)','srFOM (Alg. 3.1)','srFOM (stabilized)','interpreter','latex','FontSize',13);
xlabel('problem','FontSize',13)
ylabel('relative error','FontSize',13)
mypdf('fig/inv_sqrt_exact_error_curves',.66,1.4)
hold off;
shg

%%
%Plot the Arnoldi cycle length needed for each problem to converge.
figure
plot(fom_m,'-','LineWidth',2);
grid on;
hold on;
plot(sfom_m,'--','LineWidth',2);
plot(rfom_m,'V-','LineWidth',2);
semilogy(srfom_sRR_m,'V--','LineWidth',2);
legend('FOM','sFOM', 'rFOM (closed)','srFOM','interpreter','latex','FontSize',13);
xlabel('problem','FontSize',13)
ylabel('$m$','interpreter','latex','FontSize',13);
mypdf('fig/inv_sqrt_adaptive',.66,1.4)
shg




