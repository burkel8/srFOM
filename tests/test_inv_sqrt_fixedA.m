% test_inv_sqrt_fixedA.m

% This file evaluates a sequence of vectors of the form f(A)b where f is
% the inverse square root function. In this variant, A*U is not updated.

% The sequence of vectors is evaluated using the following 4 methods:

% fom: The standard fom approximation
% rfom: The recycled fom presented in [1]
% srfom: The sketched and recycled fom presented in [1].
% sfom: The sketched fom presented in [2]

% [1] L. Burke, S. Güttel. - Krylov Subspace Recycling With Randomized Sketching For Matrix Functions,
% arXiv :2308.02290 [math.NA] (2023)

% [2] S. Güttel, M. Schweitzer - Randomized sketching for Krylov approximations of
% large-scale matrix functions, arXiv : arXiv:2208.11447 [math.NA] (2022)

clear all, clc, close all
addpath(genpath('../'));
mydefaults

% The maximum number of iterations allowed for each method.
max_it = 400;

% Boolean variable to descide if the Arnoldi vectors in fom and
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
% orthogonalized against the previous t vectors
t = 2;

% A matrix whos columns span the recycling subspace (default empty)
U = [];

% The number of f(A)b vectors in the sequence to evaluate
num_problems = 30;

% Sketching parameter (number of rows of sketched matrix S)
s = 400;

% "Strength" of matrix perturbation (default 0, special
% case when matrix remains fixed throughout the sequence )
pert = 0;

% Monitor error every d iterations
d = 10;

% err_monitor set to "estimate" or "exact" to determine if the error should
% be estimated or computed exactly.
err_monitor = "estimate";

% runs parameter determines the number of times to run a given
% experiment. Only used for more robust timings. Default is set to 1,
% but if interested in timings, we recommend setting to 10 or higher.
runs = 1;

% If data is precomputed, load it
if isfile("qcdsqrt-8.mat") == 1

    fprintf("\n Loading Data\n");
    load qcdsqrt-8.mat

else % Or else, generate it

    fprintf("\nGenerating a sequence of %d matrices, vectors and exact solutions \n", num_problems);
    fprintf("\n This may take a while! \n");
    load("../data/conf6_0-4x4-30.mat");
    A = Problem.A;
    n = size(A,1);
    A = A + 6.0777*speye(n);
    rng('default')
    B = randn(n,num_problems);
    pert = 1e-8;
    for j = 1:num_problems
        AA{j} = A;
        E{j} = sqrtm(full(A))\B;
        A = A + pert*sprandn(A);
        fprintf("\n Problem %d data generated!", j);
    end
    save qcdsqrt-8 AA B E
    fprintf("\n Finished generating data!\n \n");

end

n = size(B,1);
param.n = n;
param.max_it = max_it;
param.reorth = reorth;
param.tol = tol;
param.svd_tol = svd_tol;
param.k = k; 
param.t = t;
param.U = U;
param.pert = pert;
param.d = d; 
param.err_monitor = err_monitor;
param.s = s;
param.hS = srft(n,s);
param.fm = @(X,v) sqrtm(full(X))\v;

% Construct subspace embedding matrix
hS = srft(param.n,s);

% Preallocate vectors
% Vectors to store errors.
fom_err = zeros(1,num_problems);
sfom_err = zeros(1,num_problems);
rfom_err = zeros(1,num_problems);
srfom_err = zeros(1,num_problems);
srfomstab_err = zeros(1,num_problems);

% Vectors to store Arnoldi cycle lengths.
fom_m  = zeros(1,num_problems);
sfom_m =  zeros(1,num_problems);
rfom_m = zeros(1,num_problems);
srfom_m = zeros(1,num_problems);
srfomstab_m = zeros(1,num_problems);

% Vectors to store matrix vector products.
fom_mv =  zeros(1,num_problems);
sfom_mv =  zeros(1,num_problems);
rfom_mv = zeros(1,num_problems);
srfom_mv = zeros(1,num_problems);
srfomstab_mv = zeros(1,num_problems);

% Vectors to store inner products.
fom_ip =  zeros(1,num_problems);
sfom_ip = zeros(1,num_problems);
rfom_ip =  zeros(1,num_problems);
srfom_ip =  zeros(1,num_problems);
srfomstab_ip =  zeros(1,num_problems);

% Vectors to store number of vector sketches.
sfom_sv =   zeros(1,num_problems);
srfom_sv = zeros(1,num_problems);
srfomstab_sv =  zeros(1,num_problems);

% Test FOM
fprintf("\n ### FOM ### \n");
tic
for run = 1:runs
    for i = 1:num_problems
        A = AA{i};
        b = B(:,i);
        param.exact = E{i}(:,i);
        out = fom(A,b,param);
        fom_err(i) = norm(out.approx - param.exact)/norm(param.exact);
        fom_m(i) = out.m;
        fom_mv(i) = out.mv;
        fom_ip(i) = out.ip;
    end
end
toc/runs
fprintf('Total matvecs: %5d - dotprods: %5d\n',sum(fom_mv),sum(fom_ip))

% Test sketched FOM
fprintf("\n ### sFOM ### \n");
tic
for run = 1:runs
    for i = 1:num_problems
        A = AA{i};
        b = B(:,i);
        param.exact = E{i}(:,i);
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

% Test recycled FOM
fprintf("\n ### rFOM ### \n");
tic
for run = 1:runs
    param.U = []; param.AU = [];
    for i = 1:num_problems
        A = AA{i};
        b = B(:,i);
        param.exact = E{i}(:,i);
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

% Test sketched-recycled FOM
fprintf("\n ### srFOM ### \n");
tic
for run = 1:runs
    param.U = []; param.SU = []; param.SAU = [];
    for i = 1:num_problems
        A = AA{i};
        b = B(:,i);
        param.exact = E{i}(:,i);
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

% Test stabilized version of sketched-recycled FOM
fprintf("\n ### srFOM (stab) ### \n");
tic
for run = 1:runs
    param.U = []; param.SU = []; param.SAU = [];
    for i = 1:num_problems
        A = AA{i};
        b = B(:,i);
        param.exact = E{i}(:,i);
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
grid on
hold on
semilogy(sfom_err,'--');
semilogy(rfom_err,'V-');
semilogy(srfom_err,'+--');
semilogy(srfomstab_err,'s--');
legend('FOM','sFOM','rFOM','srFOM','srFOM (stabilized)');
xlabel('problem')
ylabel('relative error')
title("inverse square root, fixed reltol")
ylim([1e-12,1e-8])

% Plot the Arnoldi cycle length needed for each problem to converge.
figure
plot(fom_m,'-');
grid on;
hold on;
plot(sfom_m,'--');
plot(rfom_m,'V-');
semilogy(srfom_m,'+--');
semilogy(srfomstab_m,'s--');
legend('FOM','sFOM', 'rFOM','srFOM','srFOM (stabilized)');
xlabel('problem')
ylabel('m');
ylim([80,150])
title("inverse square root, fixed reltol")

