% FUNCTION: out = SKETCHED_RECYCLED_FOM(A,b,param)
% A function which computes the sketched and recycled FOM approximation to f(A)b 

% INPUT:  A   The matrix  
%         b   The vector
%        param An input struct with the following fields
%                 
%        param.max_it   The maximum number of Arnoldi iterations
%        param.n        The dimension of A
%        param.reorth   Re-orthogonalization parameter
%        param.fm       fm = @(A,b) f(A)*b
%        param.tol      Convergence tolerance
%        param.U        Basis for recycling subspace
%        param.k        Dimension of recycling subspace
%        param.t        Truncated Arnoldi truncation parameter
%        param.hS       Subspace embedding matrix
%        param.s        Number of rows of subspace embedding matrix

% OUTPUT: out           An output struct with the following fields
%
%         out.m         Number of Arnoldi iterations executed 
%         out.approx    Approximation to f(A)b
%         out.err       Vector storing exact relative errors at each
%                       iteration
%         out.U         Updated recycling subspace

function [out] = sketched_recycled_fom(A,b,param)

   max_it = param.max_it;
   n = param.n;
   fm = param.fm;
   exact = param.exact;
   tol = param.tol;
   hS = param.hS;
   t = param.t;
   U = param.U;
   k = param.k;

   V = zeros(n,max_it);
   H = zeros(max_it+1,max_it);
   err = zeros(1,max_it);

   if isempty(U)
      SAW = [];
      SW = [];
   else
      SW = hS(U);
      AU = A*U;
      SAW = hS(AU);
   end

    % Arnoldi for (A,b)
    V(:,1) = b/norm(b); 
    for j = 1:max_it
        w = A*V(:,j);
        SAW = [SAW, hS(w)];

        for i = max(j-t+1,1):j
            H(i,j) = V(:,i)'*w;
            w = w - V(:,i)*H(i,j);
        end
        H(j+1,j) = norm(w);
        V(:,j+1) = w/H(j+1,j);
        W = [ U, V(:,1:j) ]; 
 
    % augmented sketched Arnoldi approx
       SW = [SW hS(V(:,j))];
       Sb = hS(b);

        [Q,R] = qr(SW,0);
        approx = W*(R\fm(Q'*SAW/R, Q'*Sb));

        err(j) = norm(exact - approx)/norm(exact);

        if err(j) < tol
              fprintf("\n Early convergence at iteration %d \n", j);
              out.m = j;
              out.approx = approx;
              out.err = err(:,1:j);

              if param.recycle_method == "sRR"
               H = R\(Q'*SAW);  % = pinv(SW)*SAW
               [X,T] = schur(H);
               ritz = ordeig(T);
               [~,ind] = sort(abs(ritz),'ascend');
               select = false(length(ritz),1);
               select(ind(1:k)) = 1;
               [X,T] = ordschur(X,T,select);  % H = X*T*X'
               if T(k+1,k)~=0   % don't tear apart 2x2 diagonal blocks of Schur factor
                keep = k+1;
              else 
               keep = k;
                end
                out.U = W*X(:,1:keep);
              elseif param.recycle_method == "stabsRR"
            
                %% More stable version 
               [UU,Sig,VV]=svd(SW,"econ");
               H = UU'*SAW*VV;
               [harmVecs, harmVals] = eig(H,Sig);
               harmVals = diag(harmVals);

               [~,iperm] = sort(abs(harmVals),'ascend');

                P = zeros(size(W,2),k);
                for i=1:k
                 P(:,i) = harmVecs(:,iperm(i));
               end
                out.U = W*VV*P;
               end
              return;
            end
    end

    if param.recycle_method == "sRR"

    % get updated Ritz vectors using sketched Rayleigh-Ritz
    H = R\(Q'*SAW);  % = pinv(SW)*SAW
    [X,T] = schur(H);
    ritz = ordeig(T);
    [~,ind] = sort(abs(ritz),'ascend');
    select = false(length(ritz),1);
    select(ind(1:k)) = 1;
    [X,T] = ordschur(X,T,select);  % H = X*T*X'
    if T(k+1,k)~=0   % don't tear apart 2x2 diagonal blocks of Schur factor
        keep = k+1;
    else 
        keep = k;
    end
    out.U = W*X(:,1:keep);

    elseif param.recycle_method == "stabsRR"

     %% More stable version 
    [UU,Sig,VV]=svd(SW,"econ");
    H = UU'*SAW*VV;
    [harmVecs, harmVals] = eig(H,Sig);
    harmVals = diag(harmVals);

    [~,iperm] = sort(abs(harmVals),'ascend');

    P = zeros(size(VV,2),k);
    for i=1:k
        P(:,i) = harmVecs(:,iperm(i));
    end
    out.U = W*VV*P;

    end

     out.m = j;
     out.approx = approx;
     out.err = err(:,1:j);
 end