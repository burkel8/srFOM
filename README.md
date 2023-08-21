# Sketched-and-recycled FOM (srFOM) 

A library that implements recycled FOM (rFOM) and sketched-and-recycled FOM (srFOM) proposed in [1] for sequences of matrix functions $f(A^{(i)})\mathbf{b}^{(i)}$ with slowly changing matrices $A^{(i)}$ and/or vectors $\mathbf{b}^{(i)}$.

## Tests
All test scripts used to generate the plots in [1] can be found in the "tests" folder. These files are:

`test_quad_v_closed.m`: compares the closed-form formulation of recycled FOM against the quadrature-based implementation 
presented in [2].

`test_inv_sqrt_fixed_m.m`: runs the inverse square root example, and generates a plot of the error for each method  obtained after a fixed number of Arnoldi iterations.

`test_inv_sqrt_fixed_reltol.m`: runs the inverse square root example, and generates a plot of the Arnoldi cycle length required for each method to converge.

`test_exp.m`: runs the matrix exponential example.

`test_inv.m`: runs the inverse function example.

## References
[1] L. Burke and S. Güttel. Krylov subspace recycling with randomized sketching for matrix functions, [arXiv:2308.02290](https://arxiv.org/abs/2308.02290),  2023.

[2]  L. Burke, A. Frommer, G. Ramirez-Hidalgo, and K. M. Soodhalter. Krylov subspace recycling for matrix functions,
[arXiv:2209.14163](https://arxiv.org/abs/2209.14163), 2022.

