function out = recycled_fom(A,b,param)

   max_it = param.max_it;
   n = param.n;
   fm = param.fm;
   reorth = param.reorth;
   exact = param.exact;
   tol = param.tol;
   U = param.U;
   k = param.k;

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

        

    % augment subspace with U
    W = [ V(:,1:j), U ]; 

    % Gram-Schmidt for U against V
    for l = j+1:size(W,2)
        w = W(:,l);
        for reo = 0:reorth
            for i = 1:l-1
                h = W(:,i)'*w;
                w = w - W(:,i)*h;
            end
        end
        w = w/norm(w);
        W(:,l) = w;
    end
    
   % fprintf('  orthogonality of augmented W: %0.2e\n', norm(W'*W - eye(size(W,2))))
     
    
    if isempty(U)
        HH = H(1:j,1:j);
    else
        AU = A*W(:,j+1:end);
        em = zeros(j,1); em(j) = 1;
        HH = [ H(1:j,1:j) ; 
               H(j+1,j)*(W(:,j+1:end)'*V(:,j+1))*em' ];
        HH = [ HH , W'*AU ];
    end


    % augmented Arnoldi approx
    approx = W*fm(HH, norm(b)*eye(size(W,2),1)); 
    err(j) = norm(exact - approx)/norm(exact);
   % fprintf('  error of augmented approximt: %0.2e\n', err(j))

        if err(j) < tol
              fprintf("\n Early convergence at iteration %d \n", j);
              out.m = j;
              out.approx = approx;
              out.err = err(:,1:j);

              % get updated Ritz vectors
              [X,T] = schur(HH);
              ritz = ordeig(T);
              [~,ind] = sort(abs(ritz),'ascend');
              select = false(length(ritz),1);
              select(ind(1:min(k,j))) = 1;
              [X,T] = ordschur(X,T,select);  % H = X*T*X'

              if T(min(k,j)+1,min(k,j))~=0   % don't tear apart 2x2 diagonal blocks of Schur factor
                 keep = min(k,j)+1;
             else 
                 keep = min(k,j);
             end

              out.U = W*X(:,1:keep);
             return;
        end
    end

    
    % get updated Ritz vectors
    [X,T] = schur(HH);
    ritz = ordeig(T);
    [~,ind] = sort(abs(ritz),'ascend');
    select = false(length(ritz),1);
    select(ind(1:min(k,j))) = 1;
    [X,T] = ordschur(X,T,select);  % H = X*T*X'

    if T(min(k,j)+1,min(k,j))~=0   % don't tear apart 2x2 diagonal blocks of Schur factor
       keep = min(k,j)+1;
    else 
       keep = min(k,j);
    end

    out.U = W*X(:,1:keep);


     out.m = j;
     out.approx = approx;
     out.err = err(:,1:j);
 end