function [X, Out] = My_GSGKAK(A,B,opts)
% Efficient Gram-Schmidt-based GKATK for solving tensor linear system
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
% 
%
%
% Coded by Yijie Wang, Beihang University, yjwangmath@gmail.com

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

%%%% setting the tolerance for checking the values of P_{\pi_k}(X^k)-X^k
if (flag && isfield(opts,'TOL_delta'))
    TOL_delta=opts.TOL_delta;
else
    TOL_delta=10^-24;
end

%%%% determining what to use as the stopping rule
if (flag && isfield(opts,'X_star'))
    X_star = opts.X_star;
    normXstar2 = norm(X_star,'fro')^2;
    strategy = 1;
else
    strategy = 0;
     normB=norm(B,'fro')^2+1e-2;
end

%%%% setting permutation
if (flag && isfield(opts,'permutation'))
    permutation = opts.permutation;
else
    permutation = 'RR';
end

%%%% setting tau
if (flag&&isfield(opts,'tau'))
    tau = opts.tau;
else
    tau = 10;
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

%% executing GS-GKATK
if isinf(tau)
    tM = max(Max_iter-1, 1);
else
    tM = max(tau-1, 0);
end

stopc = 0;
Q_list = zeros(l, p*tM, n); 
q_count = 0;                    
Q_unnorm = [];   
Q_norm = NaN;    
perm = [];

%for k = 0 : Max_iter-1
while ~stopc
    tic;
    iter = iter + 1;
    % --------- the first step ----------
    if iter == 1
        %% P_{\pi_0}
        [pi_k, perm] = get_permutation(m, iter, permutation, perm);
        rho = 0;
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
            X_fft = X_fft - tmp;
            % --- X - A^{\dagger} * tmp ---
            % size: (l × p × n)
            diff_norm2 = norm(tmp, 'fro')^2 / n;
            rho = rho + diff_norm2;
        end
        X_tmp = ifft(X_fft, [], 3, 'symmetric');
        Dk = X_tmp - X;
        
        %% Q_0 = D_0
        Q_unnorm = Dk;
        Q_norm   = norm(Q_unnorm,'fro');
        D_norm   = norm(Dk,'fro')^2;

        gamma = 0.5 * (rho + D_norm);
        lambda = gamma / (Q_norm^2);
        
        %% compute X
        X = X + lambda * Q_unnorm;
        
        %% stopping rule (iter=1)
        
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
        
        times(iter+1) = toc;
    end
    if iter == 1 && stopc == 0
        tic;
        iter = 2;
    end
    
    %% update the orthogonal basis
    Qk = Q_unnorm / Q_norm;
    
    if q_count < tM
        q_count = q_count + 1;
        slot = q_count;
    else
        slot = mod(iter-2, tM) + 1;
    end

    Q_list(:, (slot-1)*p+1:slot*p, :) = Qk;

    %% --------- iter >= 1 ----------
    
    %% P_{\pi_k}
    [pi_k, perm] = get_permutation(m, iter, permutation, perm);
    rho = 0;
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
        X_fft = X_fft - tmp;
        % --- X - A^{\dagger} * tmp ---
        % size: (l × p × n)
        diff_norm2 = norm(tmp, 'fro')^2 / n;
        rho = rho + diff_norm2;
    end
    X_tmp = ifft(X_fft, [], 3, 'symmetric');

    Dk = X_tmp - X;
    D_norm = norm(Dk,'fro')^2;
    
    %% stopping rule (P_{\pi_k}(X^k)-X^k=0)
    if D_norm < TOL_delta 
        if strategy
            err = norm(X_star - X,'fro')^2 / normXstar2;
        else
            %%%% Note that we do not use this stopping rule during our test
            err = norm(tprod(A,X)-B, 'fro')^2/normB;
        end
        errors(iter+1) = err;
        times(iter+1) = toc;
        rse = norm(X - X_star,'fro')^2 / norm(X_star, 'fro')^2;
        fprintf('D_norm_square=%.2e, RSE=%.2e', D_norm, rse);
        stopc = 1;
    end

    %% update the orthogonal basis
    alpha = zeros(q_count,1);
    for j = 1:q_count
        Qj = Q_list(:, (j-1)*p+1:j*p, :);
        alpha(j) = innerp(Qj, Dk);
    end
    
    % linear combination
    temp = zeros(l,p,n);
    for j = 1:q_count
        Qj = Q_list(:, (j-1)*p+1:j*p, :);
        temp = temp + alpha(j) * Qj;
    end
    Q_unnorm = Dk - temp;
    Q_norm   = norm(Q_unnorm, 'fro');

    %% update X^k
    gamma = 0.5 * (rho + D_norm);
    lambda = gamma / (Q_norm^2);

    X = X + lambda * Q_unnorm;

    
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



