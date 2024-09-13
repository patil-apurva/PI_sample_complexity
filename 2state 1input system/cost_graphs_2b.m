%this code works only for 2 state, 1 input systems
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
T = 30;

traj_num = 100;

n_all = [1, 5, 10, 50, 10^2, 500, 10^3, 5000, 10^4, 5*10^4, 10^5, 5*10^5];
% n_all = [500, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 5*10^5, 10^6, 5*10^6, 10^7, 5*10^7, 10^8];
% n_all = [500, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 5*10^5, 10^6];

costs_all = [];

figure(6)
hold on
set(gca, 'FontName', 'Arial', 'FontSize', 25)
xlabel('Log $n$', 'Interpreter','latex', 'FontSize', 30); ylabel('LQR cost', 'Interpreter','latex','FontSize', 30); 
set(gca,'LineWidth',1)
ax = gca;
ax.LineWidth = 1;
ax.Color = 'w';


for n = n_all
    n
    cost = 0; % cost of PI

    for j = 1:traj_num %to find the PI cost
        x_t = x_0;
%         j
        for t = 0:T-1

            ui_t_all = S_hat*randn(1, n, 'gpuArray'); %u_t from reference policy

            S_tau_all = arrayfun(@simulateMC, ui_t_all, x_t(1), x_t(2), S_hat, t, T, M, A_t(1,1), A_t(1,2), A_t(2,1), A_t(2,2), B_t(1), B_t(2)); %an array that stores S(tau) of each sample path starting at time t and state xt

            ri_all = gather(exp(-S_tau_all/lambda)); 

            ui_t_all_arr = gather(ui_t_all); 

            Ehat_ru = (ui_t_all_arr*(ri_all'))/n; 

            Ehat_r = sum(ri_all)/n; 

             u_t = Ehat_ru/Ehat_r; %optimal u_t at the current time step

             cost = cost + 0.5*(x_t.')*M_t*x_t + 0.5*(u_t.')*N_t*u_t; %cost at the current time step and the current trajectory

             w_t = mvnrnd([0, 0], Omega_t).';
             x_t = A_t*x_t + B_t*u_t + w_t;
        end
            cost = cost + 0.5*(x_t.')*M_t*x_t;   
    end
    costs_all = [costs_all, cost/traj_num];
end

log_n_all = log10(n_all);

plot(log_n_all, costs_all, 'b', 'LineWidth',2)


Theta_t = M_t; %terminal Theta, parameter for Riccati equation
kappa_t = 0; %terminal kappa, parameter for Riccati equation

for t=T-1:-1:0
    Theta_tp1 = Theta_t;
    kappa_tp1 = kappa_t;
    
    Theta_t = A_t'*Theta_tp1*A_t + M_t - A_t'*Theta_tp1*B_t/(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
    
    kappa_t = kappa_tp1 + 0.5*trace(Omega_t*Theta_tp1);
end

J0 = kappa_t + 0.5*(x_0')*Theta_t*x_0;

dummy = J0*ones(length(n_all), 1);

plot(log_n_all, dummy, '--r', 'LineWidth',2)

legend('PI', 'Analytical')
