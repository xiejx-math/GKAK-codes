% This matlab file computes the average PSNR across all frontal slices 
% based on MATLAB's built-in psnr function.
function per_psnr = PER_PSNR(A, ref)
    [~,~,n] = size(ref);
    sum_psnr = 0;
    for i = 1:n
        sum_psnr = sum_psnr + psnr(A(:,:,i), ref(:,:,i), 255);
    end
    per_psnr = sum_psnr / n;
end