%Function to compute approximation to f(A)b when f is the inverse square root function
% using implementation 2 of r(FOM)^2. The quadrature rule used is outlined
% in section 7 of the preprint.

%Inputs:
% A struct p with the following attributes
%         b - vector b for which we want to approximate f(A)b
%        V,H - basis of Krylov subspace and hessenberg matrix H constructed
%        from Arnoldi process to build basis for Km(A,b)
%        m - dimension of Krylov subspace
%        k - dimension of recycling subspace
%        U - recycling subspace
%        C - matrix C = A*U.
%        num_quad - number of quadrature points to be used
%        f_scalar - the scalar form of the matrix function f(z) , z scalar
%        f_matrix - A function f_matrix(A,b) computing the action of a matrix function of A 
%        on the vector b 

%Output: fr: Approximation to f(A)b
function out = recycled_fom_quad(A,b,param)


   max_it = param.max_it;
   n = param.n;
   reorth = param.reorth;
   exact = param.exact;
   U = param.U;
   k = param.k;
   fm = param.fm;
   num_quad_points = param.num_quad_points;
   
   V = zeros(n,max_it);
   H = zeros(max_it+1,max_it);
 

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
    end
  
    
   if isempty(U)
    approx = V(:,1:max_it)*fm(H(1:max_it,1:max_it),norm(b)*eye(j,1));
    out.err = norm(exact - approx)/norm(exact);

    [harmVecs, harmVals] = eig(eye(max_it),H(1:max_it,1:max_it));
    harmVals = diag(harmVals);
    
    [~,iperm] = sort(abs(harmVals),"descend");
    P = zeros(max_it,k);
    for i=1:k
        P(:,i) = harmVecs(:,iperm(i));
    end

    U = V(:,1:max_it)*P;
   [out.U,~] = qr(U,0);
    return;
   else

 C = A*U;
 term1 = zeros(max_it+k,1);

 % Define constant factors appearing in quadrature integration
 Vhat = [U V(:,1:max_it)];
 What = [C V(:,1:max_it)];
 G = zeros(max_it+k,max_it+k);
 G(1:k,1:k) = eye(k);
 G(k+1:max_it+k,k+1:max_it+k) = H(1:max_it,1:max_it);
 UmC = U-C;
 e = zeros(max_it,1);
 e(max_it)=1;
 hterm = H(max_it+1,max_it)*V(:,max_it+1)*e';
 VTb = Vhat'*b;
 VTW = Vhat'*What;

 R = @(zx) [zx*UmC  -hterm];
yy = @(zx) (VTW*(zx*speye(max_it+k)-G) + Vhat'*R(zx))\VTb;


%compute quadrature nodes and weights
weights = pi/num_quad_points*ones(1,num_quad_points);
t = zeros(1,num_quad_points);
for ii = 1:num_quad_points
t(ii) = cos((2*ii-1)/(2*num_quad_points) * pi);
end
tt = -1*(1-t)./(1+t);

%perform quadrature
for j = 1:num_quad_points
  yterm = yy(tt(j));
  term1 = term1 + weights(j)*(1/(1+t(j)))*yterm;
end

%Compute approximation
approx = (-2/pi)*Vhat*term1;
err = norm(exact - approx)/norm(exact);
out.err = err;

% Update recycling subspace
  UTV = U'*V(:,1:max_it);
  VTU = V(:,1:max_it)'*U;
  UTC = U'*C;
  VTC = V(:,1:max_it)'*C;
    UTVH = U'*V*H;

    A1 = zeros(k+max_it,k+max_it);
    A2 = zeros(k + max_it,k + max_it);
    A1(1:k,1:k) = eye(k);
    A1(1:k,k+1:end) = UTV;
    A1(k+1:end,1:k) = VTU;
    A1(k+1:end,k+1:end) = eye(max_it);

    A2(1:k,1:k) = UTC;
    A2(1:k,k+1:end) = UTVH;
    A2(k+1:end,1:k) = VTC;
    A2(k+1:end,k+1:end) = H(1:max_it,1:max_it);

    [harmVecs, harmVals] = eig(A1,A2);
    harmVals = diag(harmVals);
    
    [~,iperm] = sort(abs(harmVals),"descend");
    P = zeros(max_it+k,k);
    for i=1:k
        P(:,i) = harmVecs(:,iperm(i));
    end

    U = Vhat*P;
    [out.U,~] = qr(U,0);
   end
end