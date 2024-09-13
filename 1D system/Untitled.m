%this code works only for 1D systems
clear all
% close all

seed = 1245;
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
exp_num = 10;

traj_num = 100;

% n_all = [1, 5, 10, 50, 10^2, 500, 10^3, 5000, 10^4, 5*10^4, 10^5, 5*10^5];
n_all = [1, 5, 10, 50, 100, 500, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 5*10^5];
n_all_reduced = [10, 50, 100, 500, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 5*10^5];
% n_all = [5000, 10^4, 5*10^4, 10^5, 5*10^5];

costs_all = zeros(1,1);
UP = [];
eps_all = [];

Theta_t = M_t; %terminal Theta, parameter for Riccati equation
kappa_t = 0; %terminal kappa, parameter for Riccati equation

for t=T-1:-1:0
    Theta_tp1 = Theta_t;
    kappa_tp1 = kappa_t;
    
    Theta_t = A_t'*Theta_tp1*A_t + M_t - A_t'*Theta_tp1*B_t/(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
    
    kappa_t = kappa_tp1 + 0.5*trace(Omega_t*Theta_tp1);
end

J0 = kappa_t + 0.5*(x_0')*Theta_t*x_0;

figure(16)
hold on
set(gca, 'FontName', 'Arial', 'FontSize', 25)
xlabel('Log $n$', 'Interpreter','latex', 'FontSize', 30); ylabel('Log(LQR cost)', 'Interpreter','latex','FontSize', 30); 
set(gca,'LineWidth',1)
ax = gca;
ax.LineWidth = 1;
ax.Color = 'w';

for ii=1:exp_num
    itr = 1;
    for n = n_all

        n;
        cost = 0; % cost of PI
        eps = 0;

        for j = 1:traj_num %to find the PI cost
            x_t = x_0;

            for t = 0:T-1

                ui_t_all = sqrt(Omega_t_hat)*randn(1, n, 'gpuArray'); %u_t from reference policy

                S_tau_all = arrayfun(@simulateMC, ui_t_all, x_t, Omega_t_hat, t, T, M_t, A_t, B_t); %an array that stores S(tau) of each sample path starting at time t and state xt

                ri_all = gather(exp(-S_tau_all/lambda)); 

                ui_t_all_arr = gather(ui_t_all); 

                Ehat_ru = (ui_t_all_arr*(ri_all'))/n; 

                Ehat_r = sum(ri_all)/n; 

                 u_t = Ehat_ru/Ehat_r; %optimal u_t at the current time step

                 cost = cost + 0.5*(x_t')*M_t*x_t + 0.5*(u_t')*N_t*u_t; %cost at the current time step and the current trajectory

                 const1 = Ehat_r*sqrt(2*norm(Omega_t_hat)/n * log(2*input_dim/beta_t));
                 const2 = norm(Ehat_ru)*sqrt(1/(2*n)*log(2/alpha_t));
                 const3 = (Ehat_r - sqrt(1/(2*n)*log(2/alpha_t)))*Ehat_r;
                 eps_t = (const1 + const2)/const3;
                 eps_t2 = eps_t^2;
                 eps = eps + eps_t2;

                 w_t = sqrt(Omega_t)*randn;
                 x_t = A_t*x_t + B_t*u_t + w_t;
            end
                cost = cost + 0.5*(x_t')*M_t*x_t;   
        end
        if(n>=10 && ii==1)
            eps = eps/traj_num;
        %     eps_all = [eps_all, eps];
            UP = [UP, J0 + 0.8*eps];
        end
        costs_all(ii, itr) = cost/traj_num;
        itr = itr+1
    end
end

log_n_all = log10(n_all);
log_n_all_reduced = log10(n_all_reduced);

cost_mean = mean(costs_all,1);
cost_std = std(costs_all,0,1);
curve1 = cost_mean + cost_std;
curve2 = cost_mean - cost_std;
 
log_n_all2 = [log_n_all, fliplr(log_n_all)];
inBetween = [curve1, fliplr(curve2)];
grid on;
lightBlue = [91, 207, 244] / 255; 
p = fill(log_n_all2, inBetween, lightBlue);
p.EdgeColor = 'blue';
p.HandleVisibility = 'off';

plot(log_n_all, log10(cost_mean), 'b', 'LineWidth',2) %PI cost

dummy = J0*ones(length(n_all), 1);
plot(log_n_all, log10(dummy), '--r', 'LineWidth',2) %Analytical

plot(log_n_all_reduced, log10(UP), 'k', 'LineWidth',2) %UB

legend('PI', 'Analytical', 'UB')
xlim([0, 6.5]);
ylim([-0.5, 11]);
