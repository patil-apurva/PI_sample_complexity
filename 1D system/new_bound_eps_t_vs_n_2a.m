%this code works only for 1D systems
clear all
% close all

seed = 1255;
rng(seed); % Reset the CPU random number generator.
gpurng(seed); % Reset the GPU random number generator.

A_t = 0.85;
B_t = 0.10;
M_t = 3.0; %state cost wt
Omega_t = 1.5; %noise covariance
Omega_t_hat = Omega_t/(B_t^2); %noise covariance for PI
N_t = inv(Omega_t); %control cost wt
lambda = N_t/inv(Omega_t_hat);

x_0 = -3; %initial state
alpha_t = 0.0001;
beta_t = 0.0001;
T = 30;
input_dim = 1;

% n_all = [1, 5, 10, 50, 10^2, 500, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 5*10^5, 10^6, 5*10^6, 10^7, 5*10^7, 10^8];
n_all = [500, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 5*10^5, 10^6, 5*10^6, 10^7, 5*10^7, 10^8];

figure(5)
hold on
set(gca, 'FontName', 'Arial', 'FontSize', 25)
xlabel('Log $n$', 'Interpreter','latex', 'FontSize', 30); ylabel('$\epsilon_t$', 'Interpreter','latex','FontSize', 30); 
set(gca,'LineWidth',1)
ax = gca;
ax.LineWidth = 1;
ax.Color = 'w';

x_t = x_0;
eps_t_all = [];
eps_all = [];

for n = n_all
    n
    
    ui_t_all = sqrt(Omega_t_hat)*randn(1, n, 'gpuArray'); %u_t from reference policy

    S_tau_all = arrayfun(@simulateMC, ui_t_all, x_t, Omega_t_hat, 0, T, M_t, A_t, B_t); %an array that stores S(tau) of each sample path starting at time t and state xt

    ri_all = gather(exp(-S_tau_all/lambda)); 

    ui_t_all_arr = gather(ui_t_all); 

    Ehat_ru = (ui_t_all_arr*(ri_all'))/n; 

    Ehat_r = sum(ri_all)/n;

    u_t = Ehat_ru/Ehat_r; %optimal u_t at the current time step

    const1 = Ehat_r*sqrt(2*norm(Omega_t_hat)/n * log(2*input_dim/beta_t));
    const2 = norm(Ehat_ru)*sqrt(1/(2*n)*log(2/alpha_t));
    
    sum_ECs = 0;
    
    for s = 0:T
        G_s = [];
        Omega_bar_s = [];
        
        for s_prime = 0:s-1
            G_s = [G_s, A_t^(s-1-s_prime)];
            Omega_bar_s = blkdiag (Omega_bar_s, Omega_t);
        end
        
        ECs = 0.5*trace(M_t*(G_s*Omega_bar_s*G_s.' + A_t^s*x_0*x_0.'*(A_t^s).'));
        sum_ECs = sum_ECs + ECs;
    end
    
    const3 = exp(sum_ECs/lambda);
    
    eps_t = (const1 + const2)*(const3/Ehat_r);
    
   
    
%     eps_t2 = eps_t^2;
%     dummy = [dummy, eps_t2*T];
    eps_t_all = [eps_t_all, eps_t];
end 

log_n = log10(n_all);
plot(log_n, eps_t_all, '-ob', 'LineWidth', 2)
yticks([0, 20, 40, 60, 80, 100])
xticks([2 3 4 5 6 7 8])