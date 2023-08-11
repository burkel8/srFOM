# Sketched Recycled FOM (srFOM) 

A library which implements recycled FOM (rFOM), and sketeched-recycled FOM (srFOM) for sequences of matrix function applications proposed in 

[1] L. Burke, S. Güttel. - Krylov Subspace Recycling With Randomized Sketching For Matrix Functions, arXiv :2308.02290 [math.NA] (2023)

## Tests
All test run scripts used to generate the plots in [1] can be found in the Tests folder. These files are:

test_quad_v_closed: compares the closed form formulation of recycled FOM against the quadrature based implementation 
presented in [2].

test_qcd_sign1.m: runs the QCD example
test_exp.m: runs the exponential example
test_inv.m: runs the inverse function example.

## References
[2]  L. Burke, A. Frommer, G. Ramirez, K. M Soodhalter - Krylov Subspace Recycling For Matrix Functions,
arXiv :2209.14163 [math.NA] (2022)
