function A_tensor_B = tprod(A, B)
    [~, l, n] = size(A);
    [l2, ~, n2] = size(B);
    
    if l ~= l2 || n ~= n2
        error('The dimensions of the tensors do not match.');
    end
    
    A_fft = fft(A, [], 3);
    B_fft = fft(B, [], 3);
    
    C_fft = pagemtimes(A_fft, B_fft);

    A_tensor_B = ifft(C_fft, [], 3, 'symmetric');
end
