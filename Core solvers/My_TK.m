function [X, Out] = My_TK(A,B,opts)
% Tensor Kaczmarz method for solving tensor linear system
%                 A * X = B
% where * denotes the t-product
%
% Input: the measurement tensor A, the observed tensor B and opts
% opts.initial: the initial tensor X^0
% opts.TOL: the stopping rule
% opts.permutation: the permutation strategy
% .....
%
% Output: the approximate solution X and Out
% Out.errors: the relative iterative errors \|X^k-X^*\|^2/\|X^*\|^2
% Out.iter: the total number of iteration
% ....
% Based on the manuscript:
% [1] Ma, A., Molitor, D.: Randomized Kaczmarz for tensor linear systems. 
% BIT Numer. Math. 62(1), 171–194 (2022)
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

%%%% determining what to use as the stopping rule
if (flag && isfield(opts,'X_star'))
    X_star = opts.X_star;
    normXstar2 = norm(X_star,'fro')^2;
    strategy = 1;
else
    strategy = 0;
    normB=norm(B,'fro')^2+1e-2;
end

%%%% setting the permutation
if (flag && isfield(opts,'permutation'))
    permutation = opts.permutation;
else
    permutation = 'RR';
end

%%%% setting Fourier domain data
if(flag && isfield(opts,'lis_Apinv_fft'))
    lis_Apinv_fft = opts.lis_Apinv_fft;
else
    lis_Apinv = zeros(l, m, n);
    for i = 1:m
        lis_Apinv(:, i, :) = MPinv(A(i,:,:));
    end
    lis_Apinv_fft = fft(lis_Apinv,[],3);
end

if (flag && isfield(opts,'A_fft'))
    A_fft = opts.A_fft;
else
    A_fft = fft(A,[],3);
end

if (flag && isfield(opts,'B_fft'))
    B_fft = opts.B_fft;
else
    B_fft = fft(B,[],3);
end

if (flag && isfield(opts,'rngSeed'))
    rngSeed = opts.rngSeed;
else
    rngSeed = 1;
end
rng(rngSeed);

%% initialization
iter = 0;
times = [];
times = [times, 0];
errors = [];
if strategy
    errors = [errors, norm(X_star - X,'fro')^2 / normXstar2]; 
else
    errors = [errors, norm(tprod(A,X)-B,'fro')^2 / normB];
end

%% executing TK method
stopc = 0;
perm = [];
while ~stopc 
    tic;
    iter = iter + 1;
    %% P_{\pi_k}
    [pi_k, perm] = get_permutation(m, iter, permutation, perm);
    X_fft = fft(X, [], 3);
    for i = 1:m
        idx = pi_k(i);
        Aidx_fft   = A_fft(idx, :, :);      % 1 × l × n
        Apinv_i_fft = lis_Apinv_fft(:, idx, :); % l × 1 × n
        Bidx_fft   = B_fft(idx, :, :);      % 1 × p × n

        % --- tmp = A(idx) * X - B(idx) ---
        % size: (1 × p × n)
        tmp_fft = pagemtimes(Aidx_fft, X_fft) - Bidx_fft;
        tmp = pagemtimes(Apinv_i_fft, tmp_fft);
        % --- X - A^+ * tmp ---
        % size: (l × p × n)
        X_fft = X_fft - tmp;
    end
    X_tmp = ifft(X_fft, [], 3, 'symmetric');
    
    %% update X^k
    X = X_tmp;
    
    %% stopping rule
    if strategy
        err = norm(X_star - X,'fro')^2 / normXstar2;
        errors(iter+1) = err;
        if err < TOL || iter >= Max_iter
           stopc = 1;
        end
    else
        %%%% Note that we do not use this stopping rule during our test
        errors(iter+1) = norm(tprod(A,X)-B, 'fro')^2/normB;
        if  errors(iter+1) < TOL || iter >= Max_iter
            stopc = 1;
        end
    end
    %% store the CPU time
    times(iter+1) = toc;
end
%% setting Output
Out.iter       = iter;
Out.times_iter = times;
Out.times      = cumsum(times);
Out.time       = Out.times(end);
Out.errors     = errors;
Out.X          = X;