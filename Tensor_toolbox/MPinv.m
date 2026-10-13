function dagger = MPinv(A)
    [m, l, n] =size(A);
    A_fft = fft(A, [], 3);

    temp = zeros(l, m, n);
    for i = 1:n
        temp(:, :, i) = pinv(A_fft(:, :, i));
    end
    
    dagger = ifft(temp, [], 3, 'symmetric');
end