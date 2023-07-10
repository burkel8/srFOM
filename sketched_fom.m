function [out] = sketched_fom(A,b,param)

   max_it = param.max_it;
   n = param.n;
   fm = param.fm;
   exact = param.exact;
   tol = param.tol;
   hS = param.hS;
   t = param.t;

   V = zeros(n,max_it);
   H = zeros(max_it+1,max_it);
   err = zeros(1,max_it);

    % Arnoldi for (A,b)
    V(:,1) = b/norm(b); 
    for j = 1:max_it
        w = A*V(:,j);
        SAW(:,j) = hS(w);

        for i = max(j-t+1,1):j
            H(i,j) = V(:,i)'*w;
            w = w - V(:,i)*H(i,j);
        end
        H(j+1,j) = norm(w);
        V(:,j+1) = w/H(j+1,j);


       % sketched Arnoldi (sFOM)
       SW(:,j) = hS(V(:,j));  Sb = hS(b);
       [Q,R] = qr(SW(:,1:j),0);
       approx = V(:,1:j)*(R\fm(Q'*SAW(:,1:j)/R, Q'*Sb)); 
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