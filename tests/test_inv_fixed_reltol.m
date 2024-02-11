% test_inv_fixed_reltol.m

% This file evaluates a sequence of vectors of the form f(A)b where f is
% the inverse function, and plots the Arnoldi cycle length required for
% each problem to reach convergence.

% The sequence of vectors is evaluated using the following methods:

% fom: The standard fom approximation
% rfom: The recycled fom presented in [1]
% srfom: The sketched-recycled fom presented in [1].
% sfom: The sketched fom presented in [2]

% [1] L. Burke, S. Güttel. - Krylov Subspace Recycling With Randomized Sketching For Matrix Functions,
% arXiv :2308.02290 [math.NA] (2023)

% [2] S. Güttel, M. Schweitzer - Randomized sketching for Krylov approximations of
% large-scale matrix functions, arXiv : arXiv:2208.11447 [math.NA] (2022)

clear all
close all

addpath(genpath('../'));
mydefaults
rng('default')

% The maximum number of iterations allowed for each method.
max_it = 800;

% Boolean variable to descide if the Arnoldi vectors in fom and
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
% orthogonalized against the previous t vectors.
t = 2;

% Matrix whos columns span the recycling subspace (default empty)
U = [];

% The number of f(A)b vectors in the sequence to evaluate
num_problems = 30;

% sketching parameter (number of rows of sketched matrix S)
s = 900;

% Do the matrices change ? (set to 0 for no, and 1 for yes)
mat_change = 1;

% Monitor error every d iterations
d = 10;

% runs parameter determines the number of times we wish to run a given
% experiment. It is only used for more robust timings. Default is set to 1,
% but if interested in timings, we recommend setting to 10 or higher.
runs = 1;

% err_monitor set to "estimate" or "exact" to determine if the error should
% be estimated or computed exactly.
err_monitor = "exact";

% Generate Neumann matrix, shift to make it non-singular
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
param.mat_change = mat_change;
param.svd_tol = svd_tol;

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
fom_err = zeros(1,num_problems);
rfom_err = zeros(1,num_problems);
sfom_err = zeros(1,num_problems);
srfom_err = zeros(1,num_problems);
srfomstab_err = zeros(1,num_problems);

% Vectors of length num_problems which will store the total number of inner
% products performed by each method.
fom_ip = zeros(1,num_problems);
rfom_ip = zeros(1,num_problems);
sfom_ip = zeros(1,num_problems);
srfom_ip = zeros(1,num_problems);
srfomstab_ip = zeros(1,num_problems);

% Vectors of length num_problems which will store the total number of 
% vector sketches performed by the sketching algorithms.
sfom_sv = zeros(1,num_problems);
srfom_sv = zeros(1,num_problems);
srfomstab_sv = zeros(1,num_problems);

% Generate data
rng('default')
B = randn(n,num_problems);
E = A\B; % Exact solutions

fprintf("\n #### Evaluating a sequence of %d f(A)b applications ####   \n", num_problems);

% FOM
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
fprintf('Total iterations: %5d - matvecs: %5d - dotprods: %5d - time %1.2f \n', sum(fom_m),sum(fom_mv),sum(fom_ip),toc/runs )

% Sketched FOM
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

fprintf('Total iterations: %5d - matvecs: %5d - dotprods: %5d - sketches: %5d - time: %1.2f \n',sum(sfom_m),sum(sfom_mv),sum(sfom_ip),sum(sfom_sv),toc/runs)

% Recycled FOM
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

fprintf('Total iterations: %d - matvecs: %5d - dotprods: %5d - time: %1.2f\n',sum(rfom_m),sum(rfom_mv),sum(rfom_ip),toc/runs );

% Sketched-recycled FOM
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

fprintf('Total iterations %d - matvecs: %5d - dotprods: %5d - sketches: %5d - time: %1.2f \n', sum(srfom_m), sum(srfom_mv),sum(srfom_ip),sum(srfom_sv),toc/runs)

% Stabilized sketched-recycled FOM
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

fprintf('Total iterations %d - matvecs: %5d - dotprods: %5d - sketches: %5d - time: %1.2f\n', sum(srfomstab_m), sum(srfomstab_mv),sum(srfomstab_ip),sum(srfomstab_sv),toc/runs);

% Plot the Arnoldi cycle length needed for each problem to converge.
figure
plot(fom_m,'-');
grid on;
hold on;
plot(sfom_m,'--');
plot(rfom_m,'V-');
semilogy(srfom_m,'+--');
semilogy(srfomstab_m,'s--');
title("inverse function, fixed reltol");
legend('FOM','sFOM', 'rFOM','srFOM','srFOM (stab)','Orientation','horizontal');
xlabel('problem $i$', 'interpreter','latex');
ylabel('m');
ylim([50,500])
mypdf('fig/inv_fixed_reltol',.66,1.5)
shg
