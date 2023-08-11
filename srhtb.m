function S = srhtb(n, s)
	% S = srhtb(n, s)
	%* Generates a Subsampled Random Hadamard Transform, to be used as a random subspace embedding
	%
	% Input:
	% 	n is the original space size
	% 	s is the embedding dimension
	% Output:
	% 	S: function handle such that S(x) = SS*x, where SS is the s x n matrix corresponding to the randomized embedding
	%
	% Code taken from Oleg Balabanov's randKrylov but adapted to allow
    % for block columns

    D = randi([0 1], n,1)*2 - 1;
    N = 2^ceil(log(n)/log(2));
    perm = randperm(N,s);
    select = @(t,ind) t(ind,:);
    S = @(t) (1/sqrt(s)) * select(myfwht(D.*t),perm);

	% Fast Walsh Hadamard Transform
	function z = myfwht(a)
		h = 1;
		n = size(a,1);
        b = size(a,2);
		N = 2^ceil(log(n)/log(2));
		z = zeros(N,b);
        
		z(1:n,:) = a;
		while h < N
			for i = 1:2*h:N
				for j = i:(i + h - 1)
					x = z(j,:);
					y = z(j + h,:);
					z(j,:) = x + y;
					z(j + h,:) = x - y;
				end
			end
			h = 2*h;
		end
	end

end
