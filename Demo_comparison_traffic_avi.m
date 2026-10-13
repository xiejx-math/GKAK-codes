% This Matlab file is used to compare GS-GKATK-RR, GS-GKATK-SO, GS-GKATK-IS,
% tAKSHBM, TK-RR, TK-SO, and TK-IS
% based on the video data "traffic.avi"

clear; clc; close all;
%% set up

%% read video and create solution tensor
video = VideoReader('traffic.avi');

h = video.Height;   
w = video.Width;    
nFrames = video.NumFrames;  

frames = zeros(h, w, nFrames, 'double');
for i = 1:nFrames
    frame = readFrame(video);           
    grayFrame = rgb2gray(frame);        
    frames(:, :, i) = grayFrame;       
end

[l,p,n] = size(frames);
%% generate the blurring tensor A
m = l; 
band = 6;    
sigma = 1.8;  
tz = 0:band-1;
z1 = [exp(-(tz.^2)/(2*sigma^2)), zeros(1, l - band)]; 
A2 = (1/sqrt(2*pi*sigma)) * toeplitz(z1);
z2 = z1(1)*fliplr(z1(2:end)); 
A1 = (1/sqrt(2*pi*sigma)) * toeplitz(z1, z2);
A = zeros(m, l, n);
for k = 1:n
    A(:,:,k) = A1(k,1) * A2; 
end

%% construct clean tensor linear system (no added noise)
X_star = frames;
B = tprod(A, X_star);
X0_star = tprod(MPinv(A), B);

initial = zeros(l,p,n); 
rng(2025);            
TOL = 5*1e-3;
TOL_delta = 1e-10;
Full_iter = 2000;

%% precompute
lis_Apinv = zeros(l, m, n);
for i = 1:m
    lis_Apinv(:, i, :) = MPinv(A(i,:,:));
end

A_fft         = fft(A,[],3);
B_fft         = fft(B,[],3);
lis_Apinv_fft = fft(lis_Apinv,[],3);

%% parameter setup
opts_common.initial       = initial;
opts_common.X_star        = X_star;
opts_common.TOL           = TOL;
opts_common.lis_Apinv_fft = lis_Apinv_fft ;
opts_common.rngSeed       = 2025;

%% experiment params 
methods = {'GS-GKATK-RR', 'GS-GKATK-SO', 'GS-GKATK-IS', 'tAKSHBM', 'TK-RR', 'TK-SO', 'TK-IS'};
labels = {'GS-GKATK-RR', 'GS-GKATK-SO', 'GS-GKATK-IS', 'tAKSHBM', 'TK-RR', 'TK-SO', 'TK-IS'};
colors = {'blue','magenta','green','cyan','red','black','yellow'};
linestyles = {'--', '--', '--', '-', ':', ':', ':'};  

%% block partition for tAKSHBM
q = 15; % block size (number of rows in each block)
perm = randperm(m);
num_tak_blocks = ceil(m/q);
qs = cell(num_tak_blocks,1);
for i = 1:num_tak_blocks
    first_idx = (i-1)*q + 1;
    last_idx = min(i*q,m);
    qs{i} = perm(first_idx:last_idx);
end

%% run algorithms

%% GSGKATK-RR
opts_GSGKATK_RR = opts_common;
opts_GSGKATK_RR.lis_Apinv_fft=lis_Apinv_fft; 
opts_GSGKATK_RR.A_fft=A_fft; 
opts_GSGKATK_RR.B_fft=B_fft;
opts_GSGKATK_RR.Max_iter=Full_iter; 
opts_GSGKATK_RR.tau=5;
opts_GSGKATK_RR.TOL_delta=TOL_delta; 
opts_GSGKATK_RR.permutation='RR';
[~, Out_GSGKATK_RR] = My_GSGKAK(A,B,opts_GSGKATK_RR);
t_GS_RR=Out_GSGKATK_RR.times(:)'; 
e_GS_RR=Out_GSGKATK_RR.errors(:)'; 
X_GSGKATK_RR=Out_GSGKATK_RR.X;
fprintf('[GSGKATK-RR] Done, full iter=%d, time=%.3f s\n', Out_GSGKATK_RR.iter, t_GS_RR(end));

%% GSGKATK-SO
opts_GSGKATK_SO = opts_common;
opts_GSGKATK_SO.lis_Apinv_fft=lis_Apinv_fft; 
opts_GSGKATK_SO.A_fft=A_fft; 
opts_GSGKATK_SO.B_fft=B_fft;
opts_GSGKATK_SO.Max_iter=Full_iter; 
opts_GSGKATK_SO.tau=5;
opts_GSGKATK_SO.TOL_delta=TOL_delta; 
opts_GSGKATK_SO.permutation='SO';
[~, Out_GSGKATK_SO] = My_GSGKAK(A,B,opts_GSGKATK_SO);
t_GS_SO=Out_GSGKATK_SO.times(:)'; 
e_GS_SO=Out_GSGKATK_SO.errors(:)'; 
X_GSGKATK_SO=Out_GSGKATK_SO.X;
fprintf('[GSGKATK-SO] Done, full iter=%d, time=%.3f s\n', Out_GSGKATK_SO.iter, t_GS_SO(end));

%% GSGKATK-IS
opts_GSGKATK_IS = opts_common;
opts_GSGKATK_IS.lis_Apinv_fft=lis_Apinv_fft; 
opts_GSGKATK_IS.A_fft=A_fft; 
opts_GSGKATK_IS.B_fft=B_fft;
opts_GSGKATK_IS.Max_iter=Full_iter; 
opts_GSGKATK_IS.tau=5;
opts_GSGKATK_IS.TOL_delta=TOL_delta; 
opts_GSGKATK_IS.permutation='IS';
[~, Out_GSGKATK_IS] = My_GSGKAK(A,B,opts_GSGKATK_IS);
t_GS_IS=Out_GSGKATK_IS.times(:)'; 
e_GS_IS=Out_GSGKATK_IS.errors(:)'; 
X_GSGKATK_IS=Out_GSGKATK_IS.X;
fprintf('[GSGKATK-IS] Done, full iter=%d, time=%.3f s\n', Out_GSGKATK_IS.iter, t_GS_IS(end));

%% tAKSHBM
opts_tAKSHBM = opts_common;
opts_tAKSHBM.Max_iter=Full_iter*num_tak_blocks;
opts_tAKSHBM.qs=qs; 
opts_tAKSHBM.tol21=1e-14; 
opts_tAKSHBM.A_fft=A_fft; 
opts_tAKSHBM.B_fft=B_fft;
[~, Out_tAKSHBM] = My_tAKSHBM(A,B,opts_tAKSHBM);
t_hb=Out_tAKSHBM.times(1:num_tak_blocks:end)';
e_hb=Out_tAKSHBM.errors(1:num_tak_blocks:end)';
X_tAKSHBM_F=Out_tAKSHBM.X;
fprintf('[tAKSHBM] Done, equivalent full iter=%d, block size=%d, blocks=%d, time=%.3f s\n', ...
    floor(Out_tAKSHBM.iter/num_tak_blocks),q,num_tak_blocks,t_hb(end));

%% TK-RR
opts_TK_RR = opts_common;
opts_TK_RR.lis_Apinv_fft=lis_Apinv_fft; 
opts_TK_RR.A_fft=A_fft; 
opts_TK_RR.B_fft=B_fft;
opts_TK_RR.Max_iter=Full_iter; 
opts_TK_RR.permutation='RR'; 
[~, Out_TK_RR] = My_TK(A,B,opts_TK_RR);
t_TK_RR=Out_TK_RR.times(:)'; 
e_TK_RR=Out_TK_RR.errors(:)'; 
X_tensor_Kaczmarz_RR=Out_TK_RR.X;
fprintf('[TK-RR] Done, full iter=%d, time=%.3f s\n', Out_TK_RR.iter, t_TK_RR(end));

%% TK-SO
opts_TK_SO = opts_common;
opts_TK_SO.lis_Apinv_fft=lis_Apinv_fft; 
opts_TK_SO.A_fft=A_fft; 
opts_TK_SO.B_fft=B_fft;
opts_TK_SO.Max_iter=Full_iter; 
opts_TK_SO.permutation='SO'; 
[~, Out_TK_SO] = My_TK(A,B,opts_TK_SO);
t_TK_SO=Out_TK_SO.times(:)'; 
e_TK_SO=Out_TK_SO.errors(:)'; 
X_tensor_Kaczmarz_SO=Out_TK_SO.X;
fprintf('[TK-SO] Done, full iter=%d, time=%.3f s\n', Out_TK_SO.iter, t_TK_SO(end));

%% TK-IS
opts_TK_IS = opts_common;
opts_TK_IS.lis_Apinv_fft=lis_Apinv_fft; 
opts_TK_IS.A_fft=A_fft; 
opts_TK_IS.B_fft=B_fft;
opts_TK_IS.Max_iter=Full_iter; 
opts_TK_IS.TOL_delta=TOL_delta;
opts_TK_IS.permutation='IS'; 
[~, Out_TK_IS] = My_TK(A,B,opts_TK_IS);
t_TK_IS=Out_TK_IS.times(:)'; 
e_TK_IS=Out_TK_IS.errors(:)'; 
X_tensor_Kaczmarz_IS=Out_TK_IS.X;
fprintf('[TK-IS] Done, full iter=%d, time=%.3f s\n', Out_TK_IS.iter, t_TK_IS(end));

%% plot full iter
set(groot, "defaultFigurePosition", [200 200 560 420])
figure; hold on; box on;
err_list = {e_GS_RR, e_GS_SO, e_GS_IS, e_hb, e_TK_RR, e_TK_SO, e_TK_IS};
for m_idx = 1:numel(methods)
    e = err_list{m_idx}; x = 0:length(e)-1;
    semilogy(x, e, colors{m_idx},'LineWidth',2,'LineStyle',linestyles{m_idx});
end
set(gca,'YScale','log'); 
ylim([1*1e-4, 1.1]);
ylabel('RSE','Interpreter','latex', 'FontWeight', 'bold', 'FontSize', 14);
xlabel('Number of full iterations','Interpreter','latex', 'FontWeight', 'bold', 'FontSize', 14);
lgd1 = legend(labels, 'Interpreter', 'latex','location', 'best','FontSize',14);
pos1 = lgd1.Position;
lgd1.Position = [pos1(1)-0.02, pos1(2), pos1(3)+0.03, pos1(4)-0.02];

%% plot CPU
figure; hold on; box on;
semilogy(t_GS_RR,e_GS_RR,colors{1},'LineWidth',2,'LineStyle',linestyles{1});
semilogy(t_GS_SO,e_GS_SO,colors{2},'LineWidth',2,'LineStyle',linestyles{2});
semilogy(t_GS_IS,e_GS_IS,colors{3},'LineWidth',2,'LineStyle',linestyles{3});
semilogy(t_hb,e_hb,colors{4},'LineWidth',2,'LineStyle',linestyles{4});
semilogy(t_TK_RR,e_TK_RR,colors{5},'LineWidth',2,'LineStyle',linestyles{5});
semilogy(t_TK_SO,e_TK_SO,colors{6},'LineWidth',2,'LineStyle',linestyles{6});
semilogy(t_TK_IS,e_TK_IS,colors{7},'LineWidth',2,'LineStyle',linestyles{7});
set(gca,'YScale','log'); 
ylim([1*1e-4, 1.1]);
ylabel('RSE','Interpreter','latex', 'FontWeight', 'bold', 'FontSize', 14); 
xlabel('CPU','Interpreter','latex', 'FontWeight', 'bold', 'FontSize', 14);
lgd2 = legend(labels, 'Interpreter', 'latex','location', 'best','FontSize',14);
pos2 = lgd2.Position;
lgd2.Position = [pos2(1)-0.02, pos2(2), pos2(3)+0.03, pos2(4)-0.02];

%% SSIM & PSNR 
ssim_B = PER_SSIM(B, frames);
ssim_g = [PER_SSIM(X_GSGKATK_RR, frames), PER_SSIM(X_GSGKATK_SO, frames), PER_SSIM(X_GSGKATK_IS, frames)];
ssim_t = [PER_SSIM(X_tensor_Kaczmarz_RR, frames), PER_SSIM(X_tensor_Kaczmarz_SO, frames), PER_SSIM(X_tensor_Kaczmarz_IS, frames)];
ssim_ak = PER_SSIM(X_tAKSHBM_F, frames);

psnr_B = PER_PSNR(B, frames);
psnr_g = [PER_PSNR(X_GSGKATK_RR, frames), PER_PSNR(X_GSGKATK_SO, frames), PER_PSNR(X_GSGKATK_IS, frames)];
psnr_t = [PER_PSNR(X_tensor_Kaczmarz_RR, frames), PER_PSNR(X_tensor_Kaczmarz_SO, frames), PER_PSNR(X_tensor_Kaczmarz_IS, frames)];
psnr_ak = PER_PSNR(X_tAKSHBM_F, frames);

fprintf('\n==================== SSIM/PSNR Results ====================\n');
fprintf('Blurred:\t\t SSIM=%.4f\t PSNR=%.4f\n', ssim_B, psnr_B);
fprintf('GS-GKATK-RR:\t\t SSIM=%.4f\t PSNR=%.4f\n', ssim_g(1), psnr_g(1));
fprintf('GS-GKATK-SO:\t\t SSIM=%.4f\t PSNR=%.4f\n', ssim_g(2), psnr_g(2));
fprintf('GS-GKATK-IS:\t\t SSIM=%.4f\t PSNR=%.4f\n', ssim_g(3), psnr_g(3));
fprintf('tAKSHBM:\t\t SSIM=%.4f\t PSNR=%.4f\n', ssim_ak, psnr_ak);
fprintf('TK-RR:\t\t\t SSIM=%.4f\t PSNR=%.4f\n', ssim_t(1), psnr_t(1));
fprintf('TK-SO:\t\t\t SSIM=%.4f\t PSNR=%.4f\n', ssim_t(2), psnr_t(2));  
fprintf('TK-IS:\t\t\t SSIM=%.4f\t PSNR=%.4f\n', ssim_t(3), psnr_t(3)); 
fprintf('=============================================================\n');

%% reconstruction visualization 
figure('Position', [50, 50, 1300, 250]);  
set(gcf, 'color', 'w'); 
slices_to_show = 72;
num_slices = length(slices_to_show);  

methods_show = {'Original','Blurred','tAKSHBM','GS-GKATK-SO','TK-SO'};
num_methods = length(methods_show);        

% Reserve space for the metric titles and method labels.  At this figure
% size, each image axes is about 224-by-170 pixels (close to the original
% 4:3 ratio), while retaining enough room for the text above and below.
margin_left=0.05; margin_right=0.05; margin_top=0.17; margin_bottom=0.15;
gap_h=0.010; gap_v=0.030;
subplot_w = (1-margin_left-margin_right-(num_methods-1)*gap_h)/num_methods;
subplot_h = (1-margin_top-margin_bottom-(num_slices-1)*gap_v)/num_slices;

%%
for i = 1:num_slices
    idx = slices_to_show(i);
    img_ori = squeeze(frames(:,:,idx));
    img_blur = squeeze(B(:,:,idx));
    img_ak = squeeze(X_tAKSHBM_F(:,:,idx));
    img_g = squeeze(X_GSGKATK_SO(:,:,idx));
    img_t = squeeze(X_tensor_Kaczmarz_SO(:,:,idx));
    img_all = {img_ori, img_blur, img_ak, img_g, img_t};
    
    for j = 1:num_methods
        pos_x = margin_left + (j-1)*(subplot_w+gap_h);
        pos_y = 1 - margin_top - (i-1)*(subplot_h+gap_v) - subplot_h;
        axes('Position', [pos_x, pos_y, subplot_w, subplot_h]);

        if j == 1  % Original image
            title_str = sprintf('Frame %d', slices_to_show(i));
            imagesc(img_all{j});
        elseif j == 2  % Blurred image
            ssim_val = ssim(img_all{j}, img_ori,'DynamicRange', 255);
            psnr_val = psnr(img_all{j}, img_ori,255);
            title_str = sprintf('SSIM: %.3f, PSNR: %.2f', ...
                ssim_val, psnr_val);
            imagesc(img_all{j});
        else  % Reconstruction methods
            ssim_val = ssim(img_all{j}, img_ori,'DynamicRange', 255);
            psnr_val = psnr(img_all{j}, img_ori,255);
            title_str = sprintf('SSIM: %.3f, PSNR: %.2f', ...
                ssim_val, psnr_val);
            imagesc(img_all{j});
        end
        
        axis tight; 
        colormap(gray); 
        axis off;
        
        title(title_str, 'FontSize', 13, 'FontWeight', 'normal','Interpreter','latex');
 
    end
end

%%
text_width = 0.08;  
text_height = 0.10;
for j = 1:num_methods
    pos_x = margin_left + (j-1)*(subplot_w+gap_h) + subplot_w/2;
    left = pos_x - text_width / 2;
    bottom = 0.045;
    annotation('textbox', [left, bottom, text_width, text_height], ...
        'String', methods_show{j}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'EdgeColor', 'none', ...
        'FontSize', 14, ...
        'Interpreter', 'latex');
end
