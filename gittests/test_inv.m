%% test_inv.m

% This file evaluates a sequence of vectors of the form f(A)b where f is
% the inverse function and the matrix A is a Poisson matrix which
% is perturbed as the sequence progresses. The right hand sides are
% randomly generated.

% The sequence of vectors is evaluated using the following 4 methods
% fom: The standard Arnoldi approximation
% rfom: The recycled fom presented in [Stefan and Liam work]
% sfom: The sketched fom presented in [Stefan and Marcel]
% srfom: The sketched and recycled fom presented in [Stefan and Liam work].

% The definiton of the test inputs is given below

clear all
close all

addpath(genpath('../'));
mydefaults
rng('default')

% The maximum number of iterations of the methods
max_it = 800;

% Boolean variable to descide if the Arnoldi vectors in fom
% rfom should be re-orthogonalized (set to 1), or not (set to 0)
% (default is 0)
reorth = 0;

% The error tolerance used to define convergence.
tol = 1e-09;

% Error tolerance of SVD decompositon
svd_tol = 1e-13;

% The dimension of the recycling subspace used by rfom and srfom
k = 30;

% Arnoldi truncation parameter for the truncated Arnoldi method
% used in sfom and srfom. Each Arnoldi vector is
% orthogonalized against the previous t vectors (default t = 3)
t = 2;

% A matrix whos columns span the recycling subspace (default empty)
U = [];

% The number of f(A)b vectors in the sequence to evaluate
num_problems = 30;

% sketching parameter (number of rows of sketched matrix S)
s = 900;

% represents "strength" of matrix perturbation (default 0, special
% case when matrix remains fixed throughout the sequence )
pert = 0;

% Compute exact error or estimate error ever d iterations
d = 10;

% err_monitor set to "estimate" or "exact" to determine if the error should
% be estimated or computed exactly.
err_monitor = "exact";

addpath(genpath('../'));

% Generate Poisson Matrix
n = 10609;
A = gallery('neumann', n) + 0.001*speye(n);

% Construct subspace embedding matrix
hS = srft(n,s);

% Construct function handle which inputs a matrix A, vector b and returns
% f(A)*b where f is the inverse function
fm = @(X,v) full(X)\v;

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
param.svd_tol = svd_tol;

% input structs for fom, rfom, srfom
fom_param = param;
rfom_param = param;
sfom_param = param;
srfom_sRR_param = param;
srfom_stabsRR_param = param;

srfomstab_param.svd_tol = svd_tol;

% Vectors of length num_problems which will store the total Arnoldi cycle
% length needed for each problem in the sequence to converge.
fom_m = zeros(1,num_problems);
rfom_m = zeros(1,num_problems);
sfom_m = zeros(1,num_problems);
srfom_m = zeros(1,num_problems);
srfomstab_m = zeros(1,num_problems);

% Vectors of length num_problems which will store the total number of matrix-vector
% products needed for each problem in the sequence to converge.
fom_mv = zeros(1,num_problems);
rfom_mv = zeros(1,num_problems);
sfom_mv = zeros(1,num_problems);
srfom_mv = zeros(1,num_problems);
srfomstab_mv = zeros(1,num_problems);

% Vectors of length num_problems which will store the final exact error
% (only meaningful if the max number of iterations were performed)
fom_err = zeros(1,num_problems);
rfom_err = zeros(1,num_problems);
sfom_err = zeros(1,num_problems);
srfom_err = zeros(1,num_problems);
srfomstab_err = zeros(1,num_problems);

rng('default')
B = randn(n,num_problems);
E = A\B; % exact solutions

runs = 1; % use 10 or higher for more robust final timings

%%
fprintf("\n ### FOM ### \n");
tic
for run = 1:runs
    for i = 1:num_problems
        b = B(:,i);
        param.exact = E(:,i);
        out = fom(A,b,param);
        fom_err(i) = norm(out.approx - param.exact)/norm(param.exact);
        fom_m(i) = out.m; 
        fom_mv(i) = out.mv; 
        fom_ip(i) = out.ip; 
    end
end
toc/runs
fprintf('Total matvecs: %5d - dotprods: %5d\n',sum(fom_mv),sum(fom_ip))

%%
fprintf("\n ### sFOM ### \n");
tic
for run = 1:runs
    for i = 1:num_problems
        b = B(:,i);
        param.exact = E(:,i);
        out = sketched_fom(A,b,param);
        sfom_err(i) = norm(out.approx - param.exact)/norm(param.exact);
        sfom_m(i) = out.m; 
        sfom_mv(i) = out.mv; 
        sfom_ip(i) = out.ip; 
        sfom_sv(i) = out.sv;
    end
end
toc/runs
fprintf('Total matvecs: %5d - dotprods: %5d - sketches: %5d\n',sum(sfom_mv),sum(sfom_ip),sum(sfom_sv))

%%
fprintf("\n ### rFOM ### \n");
tic
for run = 1:runs
    param.U = []; param.AU = [];
    for i = 1:num_problems
        b = B(:,i);
        param.exact = E(:,i);
        out = recycled_fom_closed(A,b,param);
        rfom_err(i) = norm(out.approx - param.exact)/norm(param.exact);
        rfom_m(i) = out.m;
        rfom_mv(i) = out.mv;
        rfom_ip(i) = out.ip; 
        param.U = out.U; param.AU = out.AU;
    end
end
toc/runs
fprintf('Total matvecs: %5d - dotprods: %5d\n',sum(rfom_mv),sum(rfom_ip))

%%
fprintf("\n ### srFOM ### \n");
tic
for run = 1:runs
    param.U = []; param.SU = []; param.SAU = [];
    for i = 1:num_problems
        b = B(:,i);
        param.exact = E(:,i);
        out = sketched_recycled_fom(A,b,param);
        srfom_err(i) = norm(out.approx - param.exact)/norm(param.exact);
        srfom_m(i) = out.m;
        srfom_mv(i) = out.mv;
        srfom_ip(i) = out.ip; 
        srfom_sv(i) = out.sv;
        param.U = out.U; param.SU = out.SU; param.SAU = out.SAU;
    end
end
toc/runs
fprintf('Total matvecs: %5d - dotprods: %5d - sketches: %5d\n',sum(srfom_mv),sum(srfom_ip),sum(srfom_sv))

%%
fprintf("\n ### srFOM (stab) ### \n");
tic
for run = 1:runs
    param.U = []; param.SU = []; param.SAU = [];
    for i = 1:num_problems
        b = B(:,i);
        param.exact = E(:,i);
        out = sketched_recycled_fom_stabilized(A,b,param);
        srfomstab_err(i) = norm(out.approx - param.exact)/norm(param.exact);
        srfomstab_m(i) = out.m;
        srfomstab_mv(i) = out.mv;
        srfomstab_ip(i) = out.ip; 
        srfomstab_sv(i) = out.sv; 
        param.U = out.U; param.SU = out.SU; param.SAU = out.SAU;
    end
end
toc/runs
fprintf('Total matvecs: %5d - dotprods: %5d - sketches: %5d\n',sum(srfomstab_mv),sum(srfomstab_ip),sum(srfomstab_sv))

% Plot final error of each problem in the sequence
figure
semilogy(fom_err,'-');
grid on;
hold on
semilogy(sfom_err,'--');
semilogy(rfom_err,'V-');
semilogy(srfom_err,'+--');
semilogy(srfomstab_err,'s--');
legend('FOM','sFOM','rFOM','srFOM','srFOM (stabilized)');
title("inverse function, fixed m");
xlabel('problem')
ylabel('relative error')
mypdf('fig/inv_error_curves',.66,1.4)
hold off;
shg

%Plot the Arnoldi cycle length needed for each problem to converge.
figure
plot(fom_m,'-');
grid on;
hold on;
plot(sfom_m,'--');
plot(rfom_m,'V-');
semilogy(srfom_m,'+--');
semilogy(srfomstab_m,'s--');
title("inverse function, fixed reltol");
legend('FOM','sFOM', 'rFOM','srFOM','srFOM (stabilized)','Orientation','horizontal');
xlabel('problem')
ylabel('m');
ylim([50,500])
mypdf('fig/inv_adaptive_m',.66,1.4)
shg

