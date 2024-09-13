%this code works only for 1D systems
clear all
close all

seed = 1255;
rng(seed); % Reset the CPU random number generator.
gpurng(seed); % Reset the GPU random number generator.

A_t = [0.9 -0.1;
    -0.1 0.8];
B_t =[1; 0];
M = 0.1;
N = 10;
M_t = M*eye(2); %state cost wt
N_t = N*eye(1); %control cost wt

Omega_t = [4, 0; 0, 0]; %noise covariance
Omega_t_hat = Omega_t(1,1);
S_hat = sqrt(Omega_t_hat);
lambda = N*Omega_t_hat;

x_0 = [-3; -3]; %initial state
alpha_t = 0.01;
beta_t = 0.0001;
T = 30;
input_dim = 1;

% n_all = [1, 5, 10, 50, 10^2, 500, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 5*10^5, 10^6, 5*10^6, 10^7, 5*10^7, 10^8];
n_all = [500, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 5*10^5, 10^6, 5*10^6, 10^7, 5*10^7, 10^8];

figure(5)
hold on
set(gca, 'FontName', 'Arial', 'FontSize', 25)
xlabel('Log $n$', 'Interpreter','latex', 'FontSize', 30); ylabel('$\epsilon_0$', 'Interpreter','latex','FontSize', 30); 
set(gca,'LineWidth',1)
ax = gca;
ax.LineWidth = 1;
ax.Color = 'w';
grid on;

x_t = x_0;
eps_t_all = [];
eps_all = [];

for n = n_all
    n
    
    ui_t_all = S_hat*randn(1, n, 'gpuArray');

    S_tau_all = arrayfun(@simulateMC, ui_t_all, x_t(1), x_t(2), S_hat, 0, T, M, A_t(1,1), A_t(1,2), A_t(2,1), A_t(2,2), B_t(1), B_t(2));%an array that stores S(tau) of each sample path starting at time t and state xt

    ri_all = gather(exp(-S_tau_all/lambda)); %(size: (1 X n))

    ui_t_all_arr = gather(ui_t_all); %array of ui_t_all

    Ehat_ru = (ui_t_all_arr*(ri_all'))/n; %(size: (1 X 1))

    Ehat_r = sum(ri_all)/n; %scalar
     
    u_t = Ehat_ru/Ehat_r;
    
    const1 = Ehat_r*sqrt(2*norm(Omega_t_hat)/n * log(2*input_dim/beta_t));
    const2 = norm(Ehat_ru, Inf)*sqrt(1/(2*n)*log(2/alpha_t));
    const3 = (Ehat_r - sqrt(1/(2*n)*log(2/alpha_t)))*Ehat_r;
    
    eps_t = (const1 + const2)/const3;
    
   
    
%     eps_t2 = eps_t^2;
%     dummy = [dummy, eps_t2*T];
    eps_t_all = [eps_t_all, eps_t];
end 

log_n = log10(n_all);
plot(log_n, eps_t_all, '-ob', 'LineWidth', 2)
yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
xticks([2 3 4 5 6 7 8])