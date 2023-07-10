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
rng('default')


%% %%%%%%%%% User Inputs %%%%%%%%%%%%%%
max_it = 20; % The maximum number of iterations of the methods 

reorth = 0;   % Boolean variable to descide if the Arnoldi vectors in fom 
              % rfom should be re-orthogonalized (set to 1), or not (set to 0)
              % (default is 0)

tol = 10e-15; % The error tolerance used to define convergence.

k = 3;       % The dimension of the recycling subspace used by rfom and srfom

t = 3;        % Arnoldi truncation parameter for the truncated Arnoldi method
              % used in sfom and srfom. Each Arnoldi vector is
              % orthogonalized against the previous t vectors (default t = 3)

U = [];       % A matrix whos columns span the recycling subspace (default empty)

num_problems = 30; % The number of f(A)b vectors in the sequence to evaluate

s = 100; % sketching parameter (number of rows of sketched matrix S)

eps = 0; % represents "strength" of matrix perturbation (default 0, special
         % case when matrix remains fixed throughout the sequence )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load QCD matrix
n = 1000;
A = -1000*gallery('tridiag', n);
b = rand(n,1);

%%% Other Matrices to try %%%%%
%load("data/convdiff_matrix.mat");

%load("data/cdBeispiel.mat"); A = sparse(A);
%n = size(A,1);


% Construct subspace embedding matrix
hS = srft(n,s);

% Construct function handle which inputs a matrix A, vector b and returns
% f(A)*b where f is the exponential function
fm = @(X,v) expm(0.1*full(X))*v;
ExA = expm(0.1*full(A));


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

% input structs for fom, rfom, srfom
fom_param = param;
rfom_param = param;
sfom_param = param;

% We will run the srfom twice and construct the recycling subspace U
% differently for each run. The first run uses the sketched Rayleigh-Ritz
% (sRR) based on a QR factorization
srfom_sRR_param = param;
srfom_sRR_param.recycle_method = "sRR";

% The second uses a more stabilized sRR based on an SVD decomposition 
srfom_stabsRR_param = param;
srfom_stabsRR_param.recycle_method = "stabsRR";

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
fom_err(i) = fom_out.err(fom_out.m);

% Compute the recycled FOM approximation, assign the output recycling
% subspace to be the input recycling subspace for the next problem.
fprintf("\n Computing rfom approximation .... \n");
[rfom_out] = recycled_fom_closed(A,b,rfom_param);
rfom_param.U = rfom_out.U;
rfom_m(i) = rfom_out.m;
rfom_err(i) = rfom_out.err(rfom_out.m);

% Compute sketched FOM approximation
fprintf("\n Computing sfom approximation .... \n");
[sfom_out] = sketched_fom(A,b,sfom_param);
sfom_m(i) = sfom_out.m;
sfom_err(i) = sfom_out.err(sfom_out.m);

% Compute the sketched and recycled FOM approximation, assign the output
% recycling subspace to be the input recycling subspace for the next
% problem.

% First call the method which uses sketched Rayleigh-Ritz
fprintf("\n Computing srfom (with sRR) approximation .... \n");
[srfom_sRR_out] = sketched_recycled_fom(A,b,srfom_sRR_param);
srfom_sRR_param.U = srfom_sRR_out.U;
srfom_sRR_m(i) = srfom_sRR_out.m;
srfom_sRR_err(i) = srfom_sRR_out.err(srfom_sRR_out.m);

% Then, call the method which uses the stabilized sketched Rayleigh-Ritz
fprintf("\n Computing srfom (with stabilized sRR) approximation .... \n");
[srfom_stabsRR_out] = sketched_recycled_fom(A,b,srfom_stabsRR_param);
srfom_stabsRR_param.U = srfom_stabsRR_out.U;
srfom_stabsRR_m(i) = srfom_stabsRR_out.m;
srfom_stabsRR_err(i) = srfom_stabsRR_out.err(srfom_stabsRR_out.m);

% Slowly change the matrix for next problem.
A = A + eps*sprand(A);

% Take new b to be FOM approximation.
b = fom_approx;

end

figure
semilogy(fom_err,'--','LineWidth',2);
grid on;
hold on
semilogy(sfom_err,'--s','LineWidth',2);
semilogy(rfom_err,'--','LineWidth',2);
semilogy(srfom_sRR_err,'--v','LineWidth',2);
semilogy(srfom_stabsRR_err,'--v', 'LineWidth',2);
legend('FOM','sFOM','rFOM (closed)','srFOM (sRR)','srFOM (stabilized sRR)','interpreter','latex','FontSize',13);
xlabel('problem','FontSize',13)
ylabel('relative error','FontSize',13)
mypdf('fig/exp_exact_error_curves',.66,1.4)
hold off;
pause(3)

