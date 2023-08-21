# Sketched Recycled FOM (srFOM) 

A library that implements recycled FOM (rFOM), and sketched-recycled FOM (srFOM) proposed in [1] for sequences of matrix function applications.

## Tests
All test scripts used to generate the plots in [1] can be found in the tests "folder". These files are:

test_quad_v_closed: compares the closed-form formulation of recycled FOM against the quadrature-based implementation 
presented in [2].

test_inv_sqrt_fixed_m.m: runs the inverse square root example, and generates a plot of the error for each method  obtained after a fixed number of Arnoldi iterations.

test_inv_sqrt_fixed_reltol.m: runs the inverse square root example, and generates a plot of the Arnoldi cycle length required for each method to converge.

test_exp.m: runs the exponential example.

test_inv.m: runs the inverse example.

## References
[1] L. Burke, S. Güttel. - Krylov Subspace Recycling With Randomized Sketching For Matrix Functions, arXiv :2308.02290 [math.NA] (2023)

[2]  L. Burke, A. Frommer, G. Ramirez-Hidalgo, K. M. Soodhalter - Krylov Subspace Recycling For Matrix Functions,
arXiv :2209.14163 [math.NA] (2022)
