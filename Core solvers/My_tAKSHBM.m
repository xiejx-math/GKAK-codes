function [X, Out] = My_tAKSHBM(A,B,opts)
% Tensor average Kaczmarz method based on stochastic heavy ball momentum
% technique for solving tensor linear system
%                 A * X = B
% where * denotes the t-product
%
% Input: the measurement tensor A, the observed tensor B and opts
% opts.initial: the initial tensor X^0
% opts.TOL: the stopping rule
% .....
%
% Output: the approximate solution X and Out
% Out.errors: the relative iterative errors \|X^k-X^*\|^2/\|X^*\|^2
% Out.iter: the total number of iteration
% ....
% Based on the manuscript:
% [1] Liao, Y., Li, W., Yang, D.: The accelerated tensor Kaczmarz algorithm with adaptive parameters for solving tensor systems. 
% Appl. Numer. Math. 202, 100–119 (2024)
%
%
% Coded by Yijie Wang, Beihang University, yj_wang@buaa.edu.cn

%% setting some parameters
[m,l,n] = size(A);
[~,p,~] = size(B);

flag=exist('opts');

%%%% setting the max iteration
if (flag && isfield(opts,'Max_iter'))
    Max_iter=opts.Max_iter;
else
    Max_iter=200000;
end

%%%% setting the tolerance
if (flag && isfield(opts,'TOL'))
    TOL=opts.TOL;
else
    TOL=10^(-12);
end

%%%% setting the initial point
if (flag && isfield(opts,'initial'))
    initialX=opts.initial;
else
    initialX=zeros(l,p,n);
end
X=initialX;
Xf = fft(X,[],3);

%%%% determining what to use as the stopping rule
if (flag && isfield(opts,'X_star'))
    X_star = opts.X_star;
    normXstar2 = norm(X_star,'fro')^2;
    strategy = 1;
else
    strategy = 0;
    normB=norm(B,'fro')^2+1e-2;
end

%%%% setting Fourier domain data
if (flag && isfield(opts,'A_fft'))
    A_fft = opts.A_fft;
else
    A_fft = fft(A,[],3);
end
Af = A_fft;

if (flag && isfield(opts,'B_fft'))
    B_fft = opts.B_fft;
else
    B_fft = fft(B,[],3);
end
Bf = B_fft;

%%%% setting the block partition
if (flag && isfield(opts,'qs'))
    qs = opts.qs;
else
    s = 15;
    perm = randperm(m);
    qs = cell(s,1);
    bsz = floor(m/s);
    for i = 1:s
        if i < s
            qs{i} = perm((i-1)*bsz+1:i*bsz);
        else
            qs{i} = perm((i-1)*bsz+1:end);
        end
    end
end

%%%% setting the tolerance for checking the invertible condition (21)
if (flag && isfield(opts,'tol21'))
    tol21 = opts.tol21;
else
    tol21 = 1e-12;
end

%%%% setting the random seed
if (flag && isfield(opts,'rngSeed'))
    rngSeed = opts.rngSeed;
else
    rngSeed = 1;
end
rng(rngSeed);

%% compute the distribution of block partition
s = numel(qs);
blockF2 = zeros(s,1);

for i = 1:s
    tau = qs{i};
    Ai = A(tau,:,:);
    blockF2(i) = norm(Ai, 'fro')^2;
end

cdf = cumsum(blockF2 / sum(blockF2));
sample_block = @() find(cdf >= rand(),1,'first');

%% initialization
times = [];
times = [times, 0];
errors = [];
Xkf   = Xf;
if strategy
    errors = [errors, norm(ifft(Xkf,[],3) - X_star,'fro')^2 / normXstar2];
else
    errors = [errors, norm(tprod(A,X)-B,'fro')^2 / normB];
end

%% executing tAKSHBM
stopc = 0;
%% iter = 1 
tic;

iter = 1;
ik = sample_block();
tau = qs{ik};

Rf = pagemtimes(Af(tau,:,:), Xkf) - Bf(tau,:,:);

normR2 = norm(Rf(:))^2;

if normR2 > 0
    Gf = zeros(size(Xkf));
    for j = 1:n
        Gf(:,:,j) = Af(tau,:,j)' * Rf(:,:,j);
    end

    normG2 = norm(Gf,'fro')^2;
    alpha = normR2 / max(normG2,eps);
    Xkp1f = Xkf - alpha * Gf;
else
    alpha = 0;
    Xkp1f = Xkf;
end
Xkm1f = Xkf;
Xkf   = Xkp1f;

%% stopping rule
Xk = ifft(Xkf,[],3);
if strategy
    err = norm(Xk - X_star,'fro')^2 / normXstar2;
    errors(2) = err;
    if err < TOL || 1 >= Max_iter
        Out.iter = iter;
        Out.times_iter = times;
        Out.times = cumsum(Out.times_iter);
        Out.time = Out.times(end);
        Out.errors = errors;
        Out.X = Xk;
        return
    end
else
    %%%% Note that we do not use this stopping rule during our test
    error1=norm(tprod(A,Xk)-B, 'fro')^2/normB;
    errors(2)=error1;
    if  error1<TOL || iter>=Max_iter
        return
    end
end

%% store the CPU time
times(2) = toc;

%% iter >= 2
while ~stopc
    iter = iter + 1;
    tic;

    ik = sample_block();
    tau = qs{ik};

    Rf = pagemtimes(Af(tau, :, :), Xkf) - Bf(tau,:,:); 
    normR2 = norm(Rf,'fro')^2;

    if normR2 == 0 
        Xkp1f = Xkf;
    else
        % gradient
        Gf = zeros(size(Xkf));
        for j = 1:n
            Gf(:,:,j) = Af(tau,:,j)' * Rf(:,:,j);
        end
        
        normG2 = norm(Gf,'fro')^2;
        
        %% momentum
        d = Xkf - Xkm1f;
        normd2 = norm(d,'fro')^2;

        ip = real(innerp(Gf, d));
        denom = normG2 * normd2 - ip^2;

        if normd2>0 && normG2>0 && denom>eps && denom>tol21*(normG2*normd2)
            alpha = (normd2 * normR2) / denom;
            beta  = (ip * normR2) / denom;
            Xkp1f = Xkf - alpha*Gf + beta*d;
        else
            alpha = normR2 / max(normG2,eps);
            Xkp1f = Xkf - alpha*Gf;
        end
    end

    %% update X^k
    Xkm1f = Xkf;
    Xkf   = Xkp1f;
    %% store the CPU time
    times(iter+1) = toc;
    %% stopping rule
    Xk = ifft(Xkf,[],3);
    if strategy
        errors(iter+1) = norm(Xk - X_star,'fro')^2 / normXstar2;
        if errors(iter+1) < TOL || iter >= Max_iter
            stopc = 1;
        end
    else
        %%%% Note that we do not use this stopping rule during our test
        errors(iter+1) = norm(tprod(A,Xk)-B, 'fro')^2/normB;
        if  errors(iter+1) < TOL || iter >= Max_iter
            stopc = 1;
        end
    end
    
end

%% setting Output
X = ifft(Xkf,[],3);
Out.iter = iter;
Out.times_iter = times;
Out.times = cumsum(Out.times_iter);
Out.time = Out.times(end);
Out.errors = errors;
Out.X = X;


