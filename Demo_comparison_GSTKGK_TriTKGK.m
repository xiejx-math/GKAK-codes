% This Matlab file is used to compare GS-TKGK-SO and Tri-TKGK-SO

clear; clc; close all;
                   
%% problem setup
m = 200; % number of horizontal slices of tensor A
l = 120; % number of lateral slices of tensor A
n = 3; % number of frontal slices of tensor A
p = 120; % number of lateral slices of tensor X
kappa = 1e1; % upper bound for the condition number of each frontal slice of A
r = 90; % upper bound for the rank of each frontal slice of A
Max_length = 35; % max number of full iterations
run_time = 10; % average times 

%% some vectors are used to store the computed results

CPU_GSTKGK=zeros(run_time,Max_length+1);
Error_GSTKGK=zeros(run_time,Max_length+1);

CPU_TRITKGK=zeros(run_time,Max_length+1);
Error_TRITKGK=zeros(run_time,Max_length+1);

%% run and store the numerical results
for ii=1:run_time
    rngSeed = ii;
    %% generated the matrix A
    A = zeros(m, l, n);
    for i = 1 : n
        [U,~]=qr(randn(m, r), 0);
        [V,~]=qr(randn(l, r), 0);
        D = diag(1+(kappa-1).*rand(r, 1));
        A(:, :, i) = U * D * V' ;
    end
    clear U V D
    %% generated the right-hand tensor B
    X_star = randn(l,p,n);
    B = tprod(A,X_star);
    initial = zeros(l,p,n);
    X0_star = tprod(MPinv(A), B);
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
    opts_common.X_star        = X0_star;
    opts_common.Max_iter      = Max_length;
    opts_common.TOL           = 1e-20;
    opts_common.lis_Apinv_fft = lis_Apinv_fft;
    opts_common.A_fft         = A_fft;
    opts_common.B_fft         = B_fft;
    opts_common.tau           = 5;
    opts_common.TOL_delta     = 1e-40;
    opts_common.rngSeed       = rngSeed;
    opts_common.permutation   = 'SO';
    
    opts_GSTKGK = opts_common;
    opts_TRITKGK = opts_common;
    %% run GSTKGK
    [x_GSTKGK,Out_GSTKGK]=My_GSTKGK(A,B,opts_GSTKGK);
    fprintf('[GS-TKGK-SO] Done, run=%d/%d, iter=%d, time=%.3f s\n', ...
            ii, run_time, Out_GSTKGK.iter, Out_GSTKGK.time);
    %% run TRITKGK
    [x_TRITKGK,Out_TRITKGK]=My_TriTKGK(A,B,opts_TRITKGK);
    fprintf('[Tri-TKGK-SO] Done, run=%d/%d, iter=%d, time=%.3f s\n', ...
            ii, run_time, Out_TRITKGK.iter, Out_TRITKGK.time);
    %%
    CPU_GSTKGK(ii,1:length(Out_GSTKGK.times))=Out_GSTKGK.times;
    Error_GSTKGK(ii,1:length(Out_GSTKGK.errors))=Out_GSTKGK.errors;
    
    CPU_TRITKGK(ii,1:length(Out_TRITKGK.times))=Out_TRITKGK.times;
    Error_TRITKGK(ii,1:length(Out_TRITKGK.errors))=Out_TRITKGK.errors;
    
    fprintf('Done, ii=%d\n',ii)
end
%% ========================= set groot ========================
set(groot, "defaultFigurePosition", [200 200 560 420])
set(groot, "defaultFigureWindowStyle", "normal")

%% plot the iterations
xlable=1:Max_length+1;

e1=Error_GSTKGK;
mine1=min(e1);
maxe1=max(e1);
e1q25=quantile(e1',0.25,2)';
e1q75=quantile(e1',0.75,2)';

e2=Error_TRITKGK;
mine2=min(e2);
maxe2=max(e2);
e2q25=quantile(e2',0.25,2)';
e2q75=quantile(e2',0.75,2)';

%%%
figure

h = fill([xlable  fliplr(xlable)], [mine2 fliplr(maxe2)],'blue','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [e2q25 fliplr(e2q75)],'blue','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([xlable  fliplr(xlable)], [mine1 fliplr(maxe1)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [e1q25 fliplr(e1q75)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .1)

%%%
hold on
p1=semilogy( xlable, median(e1), 'magenta', 'LineWidth', 1,...
    'LineStyle', '--','DisplayName', 'GSTKGK');
p2=semilogy( xlable, median(e2), 'blue', 'LineWidth', 1,...
    'LineStyle', '-.', 'DisplayName', 'TRITKGK');

hold on
set(gca, 'YScale', 'log')

xlim([1,Max_length]);
ylim([10^(-12),1]);
ylabel('RSE','Interpreter', 'latex')
xlabel('Number of iterations','Interpreter', 'latex')
hold on
txt=title(['$\kappa=$ ',num2str(kappa), ', $r=$ ',num2str(r)]);
set(txt, 'Interpreter', 'latex');
%legend([p1 p2],{'GS-TKGK-SO','Tri-TKGK-SO'},'Interpreter', 'latex','location', 'best')
lgd1 = legend([p1 p2],{'GS-TKGK-SO','Tri-TKGK-SO'},...
    'Interpreter', 'latex','location', 'best');
pos1 = lgd1.Position;
lgd1.Position = [pos1(1)-0.02, pos1(2), pos1(3)+0.02, pos1(4)];


%% plot CPU

xlable=0:Max_length;

y1=CPU_GSTKGK;
miny1=min(y1);
maxy1=max(y1);
y1q25=quantile(y1',0.25,2)';
y1q75=quantile(y1',0.75,2)';

y2=CPU_TRITKGK;
miny2=min(y2);
maxy2=max(y2);
y2q25=quantile(y2',0.25,2)';
y2q75=quantile(y2',0.75,2)';

figure

h = fill([median(y2)  fliplr(median(y2))], [mine2 fliplr(maxe2)],'blue','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([median(y2)  fliplr(median(y2))], [e2q25 fliplr(e2q75)],'blue','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([median(y1)  fliplr(median(y1))], [mine1 fliplr(maxe1)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .05)
hold on
h = fill([median(y1)  fliplr(median(y1))], [e1q25 fliplr(e1q75)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on
p1=semilogy( median(y1), median(e1), 'magenta', 'LineWidth', 1,...
    'LineStyle', '--','DisplayName', 'GSTKGK');
p2=semilogy( median(y2), median(e2), 'blue', 'LineWidth', 1,...
    'LineStyle', '-.', 'DisplayName', 'TRITKGK');

hold on
set(gca, 'YScale', 'log')

%xlim([0,11]);
ylim([10^(-12),1]);
ylabel('RSE','Interpreter', 'latex')
xlabel('CPU','Interpreter', 'latex')
%legend([p1 p2],{'GS-TKGK-SO','Tri-TKGK-SO'},'Interpreter', 'latex','location', 'best')

lgd2 = legend([p1 p2],{'GS-TKGK-SO','Tri-TKGK-SO'},...
    'Interpreter', 'latex','location', 'best');
pos2 = lgd2.Position;
lgd2.Position = [pos2(1)-0.02, pos2(2), pos2(3)+0.02, pos2(4)];

hold on
txt=title(['$\kappa=$ ',num2str(kappa), ', $r=$ ',num2str(r)]);
set(txt, 'Interpreter', 'latex');
