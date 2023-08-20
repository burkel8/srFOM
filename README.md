# Sketched Recycled FOM (srFOM) 

A library which implements recycled FOM (rFOM), and sketeched-recycled FOM (srFOM) proposed in [1] for sequences of matrix function applications.

## Tests
All test scripts used to generate the plots in [1] can be found in the Tests folder. These files are:

test_quad_v_closed: compares the closed form formulation of recycled FOM against the quadrature based implementation 
presented in [2].

test_inv_sqrt1.m: runs the inverse square root example while computing A*U between problems.

test_inv_sqrt2.m: runs the inverse square root example while avoiding computation of A*U between problems.

test_exp.m: runs the exponential example

test_inv.m: runs the inverse example.

## References
[1] L. Burke, S. Güttel. - Krylov Subspace Recycling With Randomized Sketching For Matrix Functions, arXiv :2308.02290 [math.NA] (2023)

[2]  L. Burke, A. Frommer, G. Ramirez, K. M Soodhalter - Krylov Subspace Recycling For Matrix Functions,
arXiv :2209.14163 [math.NA] (2022)
