% Inverse square root of shifted QCD matrix
% Timings and error plots
% In this variant, A*U is not updated


clear all, clc, close all
addpath(genpath('../'));
mydefaults


if 0   % generate data
    load("../data/conf6_0-4x4-30.mat");
    A = Problem.A;
    n = size(A,1);
    A = A + 6.0777*speye(n);
    rng('default')
    B = randn(n,30);
    pert = 1e-8;
    for j = 1:30, j
        AA{j} = A;
        E{j} = sqrtm(full(A))\B;
        A = A + pert*sprandn(A);
    end
    save qcdsqrt-8 AA B E
else     % or load it
    fprintf("loading precomputed data");
    load qcdsqrt-8.mat
end

rng('default')
param.max_it = 500;
param.reorth = 0;
param.tol = 1e-10;
param.svd_tol = 1e-14;
param.k = 30; % recycling subspace
param.t = 2;
param.U = [];
num_problems = 20;
param.pert = 0;   % keep A*U fixed!!!
param.d = 10; % compute exact error or estimate error every d iterations
param.err_monitor = "exact";
n = size(AA{1},1);
param.n = n;
s = 400;
param.s = s;
param.hS = srft(n,s);           % MUCH FASTER FOR THIS PROBLEM
param.fm = @(X,v) sqrtm(full(X))\v;

runs = 10; % use 10 or higher for more robust final timings

%%
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

%%
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

%%
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

%%
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

%%
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

%%
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
mypdf('fig/invSqrt_error_curves_fixAU',.66,1.4)
shg

%% Plot the Arnoldi cycle length needed for each problem to converge.
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
mypdf('fig/inv_sqrt_adaptive_m_fixAU',.66,1.4)
shg
