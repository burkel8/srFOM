% FUNCTION: out = FOM(A,b,param)
% A function which computes the standard Arnoldi approximation to f(A)b 

% INPUT:  A   The matrix  
%         b   The vector
%        param An input struct with the following fields
%                 
%        param.max_it   The maximum number of Arnoldi iterations
%        param.n        The dimension of A
%        param.reorth   Re-orthogonalization parameter
%        param.fm       fm = @(A,b) f(A)*b
%        param.tol      Convergence tolerance

% OUTPUT: out           An output struct with the following fields
%
%         out.m         Number of Arnoldi iterations executed 
%         out.approx    Approximation to f(A)b
%         out.err       Vector storing exact relative errors at each
%                       iteration
function out = fom(A,b,param)
   max_it = param.max_it;
   n = param.n;
   fm = param.fm;
   reorth = param.reorth;
   exact = param.exact;
   tol = param.tol;

   V = zeros(n,max_it);
   H = zeros(max_it+1,max_it);
   err = zeros(1,max_it);

    % Arnoldi for (A,b)
    V(:,1) = b/norm(b); 
    for j = 1:max_it
        w = A*V(:,j);
        for reo = 0:reorth
            for i = 1:j
                h = V(:,i)'*w;
                H(i,j) = H(i,j) + h;
                w = w - V(:,i)*h;
            end
        end
        H(j+1,j) = norm(w);
        V(:,j+1) = w/H(j+1,j);
        approx = V(:,1:j)*fm(H(1:j,1:j), norm(b)*eye(j,1));
        err(j) = norm(exact - approx)/norm(exact);

        if err(j) < tol
              fprintf("\n FOM converged at iteration %d \n", j);
              out.m = j;
              out.approx = approx;
              out.err = err(:,1:j);
              return;
        end
    end
     fprintf("\n FOM did not converge, max iterations performed \n", j);
     out.m = j;
     out.approx = approx;
     out.err = err(:,1:j);
 end