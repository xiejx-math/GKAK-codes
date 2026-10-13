% This matlab file is used to compare the performance of GS-TKGK-SO with
% different truncated parameter tau

clear; clc; close all;

%%
m = 100; % number of horizontal slices of tensor A
l = 75; % number of lateral slices of tensor A
n = 3; % number of frontal slices of tensor A
p = 75; % number of lateral slices of tensor X
kappa = 1e1; % upper bound for the condition number of each frontal slice of A
r = 75; % upper bound for the rank of each frontal slice of A
Max_length = 60; % max number of full iterations
run_time = 10; % average times 
tau_values = [1, 2, 3, 5, 10]; % candidate values for parameter tau

%% some vectors are used to store the computed results

CPU_GSTKGK_tau1=zeros(run_time,Max_length+1);
Error_GSTKGK_tau1=zeros(run_time,Max_length+1);

CPU_GSTKGK_tau2=zeros(run_time,Max_length+1);
Error_GSTKGK_tau2=zeros(run_time,Max_length+1);

CPU_GSTKGK_tau3=zeros(run_time,Max_length+1);
Error_GSTKGK_tau3=zeros(run_time,Max_length+1);

CPU_GSTKGK_tau5=zeros(run_time,Max_length+1);
Error_GSTKGK_tau5=zeros(run_time,Max_length+1);

CPU_GSTKGK_tau10=zeros(run_time,Max_length+1);
Error_GSTKGK_tau10=zeros(run_time,Max_length+1);

Lis_all_time = [];
%% run and store the numerical results
for ii=1:run_time
    %% generated the matrix A
    rngSeed = ii+2026;
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
    %% ===================== precompute (GSTKGK only) =====================
    lis_Apinv = zeros(l, m, n);
    for i = 1:m
        lis_Apinv(:, i, :) = MPinv(A(i,:,:));
    end
    A_fft         = fft(A,[],3);
    B_fft         = fft(B,[],3);
    lis_Apinv_fft = fft(lis_Apinv,[],3);

    %% parameter setup
    opts_common.initial    = initial;
    opts_common.X_star     = X0_star;
    opts_common.Max_iter   = Max_length;
    opts_common.TOL        = 1e-20;
    opts_common.lis_Apinv_fft = lis_Apinv_fft;
    opts_common.A_fft      = A_fft;
    opts_common.B_fft      = B_fft;
    opts_common.TOL_delta  = 1e-40;
    opts_common.rngSeed = rngSeed;
    opts_common.permutation = 'SO';
    
    opts_GSTKGK_tau1 = opts_common;
    opts_GSTKGK_tau2 = opts_common;
    opts_GSTKGK_tau3 = opts_common;
    opts_GSTKGK_tau5 = opts_common;
    opts_GSTKGK_tau10 = opts_common;
    opts_GSTKGK_tau1.tau = tau_values(1);
    opts_GSTKGK_tau2.tau = tau_values(2);
    opts_GSTKGK_tau3.tau = tau_values(3);
    opts_GSTKGK_tau5.tau = tau_values(4);
    opts_GSTKGK_tau10.tau = tau_values(5);
    %% run GSTKGK with with different tau
    [x_tau1,Out_tau1]=My_GSTKGK(A,B,opts_GSTKGK_tau1);
    Lis_all_time = [Lis_all_time, Out_tau1.time];
    fprintf('[t=1] Done, run=%d/%d, iter=%d, time=%.3f s\n', ...
            ii, run_time, Out_tau1.iter, Out_tau1.time);

    [x_tau2,Out_tau2]=My_GSTKGK(A,B,opts_GSTKGK_tau2);
    Lis_all_time = [Lis_all_time, Out_tau2.time];
    fprintf('[t=2] Done, run=%d/%d, iter=%d, time=%.3f s\n', ...
            ii, run_time, Out_tau2.iter, Out_tau2.time);

    [x_tau3,Out_tau3]=My_GSTKGK(A,B,opts_GSTKGK_tau3);
    Lis_all_time = [Lis_all_time, Out_tau3.time];
    fprintf('[t=3] Done, run=%d/%d, iter=%d, time=%.3f s\n', ...
            ii, run_time, Out_tau3.iter, Out_tau3.time);

    [x_tau4,Out_tau4]=My_GSTKGK(A,B,opts_GSTKGK_tau5);
    Lis_all_time = [Lis_all_time, Out_tau4.time];
    fprintf('[t=5] Done, run=%d/%d, iter=%d, time=%.3f s\n', ...
            ii, run_time, Out_tau4.iter, Out_tau4.time);

    [x_tau5,Out_tau5]=My_GSTKGK(A,B,opts_GSTKGK_tau10);
    Lis_all_time = [Lis_all_time, Out_tau5.time];
    fprintf('[t=10] Done, run=%d/%d, iter=%d, time=%.3f s\n', ...
            ii, run_time, Out_tau5.iter, Out_tau5.time);
    %%
    CPU_GSTKGK_tau1(ii,1:length(Out_tau1.times))=Out_tau1.times;
    Error_GSTKGK_tau1(ii,1:length(Out_tau1.errors))=Out_tau1.errors;
    
    CPU_GSTKGK_tau2(ii,1:length(Out_tau2.times))=Out_tau2.times;
    Error_GSTKGK_tau2(ii,1:length(Out_tau2.errors))=Out_tau2.errors;

    CPU_GSTKGK_tau3(ii,1:length(Out_tau3.times))=Out_tau3.times;
    Error_GSTKGK_tau3(ii,1:length(Out_tau3.errors))=Out_tau3.errors;

    CPU_GSTKGK_tau5(ii,1:length(Out_tau4.times))=Out_tau4.times;
    Error_GSTKGK_tau5(ii,1:length(Out_tau4.errors))=Out_tau4.errors;

    CPU_GSTKGK_tau10(ii,1:length(Out_tau5.times))=Out_tau5.times;
    Error_GSTKGK_tau10(ii,1:length(Out_tau5.errors))=Out_tau5.errors;
    
    fprintf('Done, ii=%d\n',ii)
end
%% ========================= set groot ========================
set(groot, "defaultFigurePosition", [200 200 560 420])
set(groot, "defaultFigureWindowStyle", "normal")


%% plot the iterations
xlable=1:Max_length+1;

e1=Error_GSTKGK_tau1;
mine1=min(e1);
maxe1=max(e1);
e1q25=quantile(e1',0.25,2)';
e1q75=quantile(e1',0.75,2)';

e2=Error_GSTKGK_tau2;
mine2=min(e2);
maxe2=max(e2);
e2q25=quantile(e2',0.25,2)';
e2q75=quantile(e2',0.75,2)';

e3=Error_GSTKGK_tau3;
mine3=min(e3);
maxe3=max(e3);
e3q25=quantile(e3',0.25,2)';
e3q75=quantile(e3',0.75,2)';

e4=Error_GSTKGK_tau5;
mine4=min(e4);
maxe4=max(e4);
e4q25=quantile(e4',0.25,2)';
e4q75=quantile(e4',0.75,2)';

e5=Error_GSTKGK_tau10;
mine5=min(e5);
maxe5=max(e5);
e5q25=quantile(e5',0.25,2)';
e5q75=quantile(e5',0.75,2)';

%%%
figure
h = fill([xlable  fliplr(xlable)], [mine5 fliplr(maxe5)],'magenta','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [e5q25 fliplr(e5q75)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([xlable  fliplr(xlable)], [mine4 fliplr(maxe4)],'blue','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [e4q25 fliplr(e4q75)],'blue','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([xlable  fliplr(xlable)], [mine3 fliplr(maxe3)],'red','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [e3q25 fliplr(e3q75)],'red','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([xlable  fliplr(xlable)], [mine2 fliplr(maxe2)],'green','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [e2q25 fliplr(e2q75)],'green','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([xlable  fliplr(xlable)], [mine1 fliplr(maxe1)],'black','EdgeColor', 'none');
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [e1q25 fliplr(e1q75)],'black','EdgeColor', 'none');
set(h,'facealpha', .1)

%%%
hold on
p1=semilogy( xlable, median(e1), 'black', 'LineWidth', 1,...
    'LineStyle', '--');
p2=semilogy( xlable, median(e2), 'green', 'LineWidth', 1,...
    'LineStyle', '--');
p3=semilogy( xlable, median(e3), 'red', 'LineWidth', 1,...
    'LineStyle', '--');
p4=semilogy( xlable, median(e4), 'blue', 'LineWidth', 1,...
    'LineStyle', '--');
p5=semilogy( xlable, median(e5), 'magenta', 'LineWidth', 1,...
    'LineStyle', '--');
hold on
set(gca, 'YScale', 'log')

xlim([1,Max_length]);
ylim([10^(-12),1]);
ylabel('RSE','Interpreter', 'latex')
xlabel('Number of iterations','Interpreter', 'latex')
hold on
txt=title(['$\kappa=$ ',num2str(kappa), ', $r=$ ',num2str(r)],'FontName', 'Times New Roman');
set(txt, 'Interpreter', 'latex');
legend([p1 p2 p3 p4 p5],{'$\tau=1$', '$\tau=2$', '$\tau=3$', '$\tau=5$', '$\tau=10$'},'Interpreter', 'latex','location', 'best')

%% plot CPU

xlable=0:Max_length;

y1=CPU_GSTKGK_tau1;
miny1=min(y1);
maxy1=max(y1);
y1q25=quantile(y1',0.25,2)';
y1q75=quantile(y1',0.75,2)';

y2=CPU_GSTKGK_tau2;
miny2=min(y2);
maxy2=max(y2);
y2q25=quantile(y2',0.25,2)';
y2q75=quantile(y2',0.75,2)';

y3=CPU_GSTKGK_tau3;
miny3=min(y3);
maxy3=max(y3);
y3q25=quantile(y3',0.25,2)';
y3q75=quantile(y3',0.75,2)';

y4=CPU_GSTKGK_tau5;
miny4=min(y4);
maxy4=max(y4);
y4q25=quantile(y4',0.25,2)';
y4q75=quantile(y4',0.75,2)';

y5=CPU_GSTKGK_tau10;
miny5=min(y5);
maxy5=max(y5);
y5q25=quantile(y5',0.25,2)';
y5q75=quantile(y5',0.75,2)';

med1 = median(y1);
med2 = median(y2);
med3 = median(y3);
med4 = median(y4);
med5 = median(y5);
all_cpu_data = [med1, med2, med3, med4, med5];
x_max = max(all_cpu_data(:)) * 1.02; 

figure
h = fill([median(y5)  fliplr(median(y5))], [mine5 fliplr(maxe5)],'magenta','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([median(y5)  fliplr(median(y5))], [e5q25 fliplr(e5q75)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([median(y4)  fliplr(median(y4))], [mine4 fliplr(maxe4)],'blue','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([median(y4)  fliplr(median(y4))], [e4q25 fliplr(e4q75)],'blue','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([median(y3)  fliplr(median(y3))], [mine3 fliplr(maxe3)],'red','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([median(y3)  fliplr(median(y3))], [e3q25 fliplr(e3q75)],'red','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([median(y2)  fliplr(median(y2))], [mine2 fliplr(maxe2)],'green','EdgeColor', 'none');
hold on
set(h,'facealpha', .05)
hold on
h = fill([median(y2)  fliplr(median(y2))], [e2q25 fliplr(e2q75)],'green','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on

h = fill([median(y1)  fliplr(median(y1))], [mine1 fliplr(maxe1)],'black','EdgeColor', 'none');
set(h,'facealpha', .05)
hold on
h = fill([median(y1)  fliplr(median(y1))], [e1q25 fliplr(e1q75)],'black','EdgeColor', 'none');
set(h,'facealpha', .1)

hold on
p1=semilogy(median(y1), median(e1), 'black', 'LineWidth', 1,...
    'LineStyle', '--');
p2=semilogy(median(y2), median(e2), 'green', 'LineWidth', 1,...
    'LineStyle', '--');
p3=semilogy(median(y3), median(e3), 'red', 'LineWidth', 1,...
    'LineStyle', '--');
p4=semilogy(median(y4), median(e4), 'blue', 'LineWidth', 1,...
    'LineStyle', '--');
p5=semilogy(median(y5), median(e5), 'magenta', 'LineWidth', 1,...
    'LineStyle', '--');
hold on
set(gca, 'YScale', 'log')

xlim([0,0.45]);
ylim([10^(-12),1]);
ylabel('RSE','Interpreter', 'latex')
xlabel('CPU','Interpreter', 'latex')
legend([p1 p2 p3 p4 p5],{'$\tau=1$', '$\tau=2$', '$\tau=3$', '$\tau=5$', '$\tau=10$'},'Interpreter', 'latex','location', 'best')

hold on
txt=title(['$\kappa=$ ',num2str(kappa), ', $r=$ ',num2str(r)]);
set(txt, 'Interpreter', 'latex');
