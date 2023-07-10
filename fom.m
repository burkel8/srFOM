function [out] = fom(A,b,param)

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
              fprintf("\n Early convergence at iteration %d \n", j);
              out.m = j;
              out.approx = approx;
              out.err = err(:,1:j);
              return;
        end
    end

     out.m = j;
     out.approx = approx;
     out.err = err(:,1:j);
 end