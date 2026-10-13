function A = psf_to_tproduct_operator(psf,image_size)
%PSF_TO_TPRODUCT_OPERATOR Build a t-product tensor for 2-D circular blur.
%   A = psf_to_tproduct_operator(PSF,[H W]) returns H-by-H-by-W A such
%   that, for X of size H-by-NFRAMES-by-W, squeeze(tprod(A,X)(:,j,:)) is
%   the 2-D circular convolution of squeeze(X(:,j,:)) with PSF.

if numel(image_size) ~= 2
    error('image_size must be [height width].');
end
h = image_size(1);
w = image_size(2);
[kh,kw] = size(psf);
if kh > h || kw > w
    error('The PSF cannot be larger than the image.');
end

psf = double(psf);
psf = psf / sum(psf(:));
padded = zeros(h,w);
padded(1:kh,1:kw) = psf;
padded = circshift(padded,-floor([kh,kw]/2));

% FFT along image columns gives the t-product Fourier pages. Each page is
% a circulant matrix implementing convolution along image rows.
column_spectra = fft(padded,[],2);
row_index = mod((0:h-1)'-(0:h-1),h)+1;
Af = zeros(h,h,w,'like',column_spectra);
for k = 1:w
    first_column = column_spectra(:,k);
    Af(:,:,k) = first_column(row_index);
end
A = ifft(Af,[],3,'symmetric');
end
