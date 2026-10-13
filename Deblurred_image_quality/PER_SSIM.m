% This matlab file computes the average SSIM across all frontal slices 
% based on MATLAB's built-in ssim function.
function per_ssim= PER_SSIM(A, ref)
    [~,~,n] = size(ref);
    sum_ssim = 0;
    for i = 1:n
        sum_ssim = sum_ssim + ssim(A(:,:,i), ref(:,:,i), 'DynamicRange', 255);
    end
    per_ssim = sum_ssim / n;
end