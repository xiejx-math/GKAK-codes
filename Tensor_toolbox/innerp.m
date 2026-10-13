function ip = innerp(A, B)
    ip = sum(A.*conj(B), 'all');
end