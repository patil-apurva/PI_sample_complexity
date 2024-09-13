clear all;
close all;


X_t = [-3, -2, -1, 0, 1, 2, 3];
A_t = 0.85;
B_t = 0.10;
M_t = 3.0; %state cost wt
Omega_t = 1.5; %noise covariance
Omega_t_hat = Omega_t/(B_t^2);
N_t = inv(Omega_t); %control cost wt
alpha_t = 0.001;
beta_t = 0.001;
lambda = N_t/inv(Omega_t_hat);
T = 30;

figure(1)
hold on
set(gca, 'FontName', 'Arial', 'FontSize', 20)
xlabel('$\epsilon$', 'Interpreter','latex', 'FontSize', 30); ylabel('$J_0^*$', 'Interpreter','latex','FontSize', 30); 
set(gca,'LineWidth',1)
ax = gca;
ax.LineWidth = 1;
ax.Color = 'w';


cost = zeros(length(X_t), 1);
for itr = 1:length(X_t)
   
    x_t = X_t(itr);

    cost_dummy = 0;
    n = 100000;
 
    for t=0:T-1
        
        cost_dummy = cost_dummy + 0.5*x_t'*M_t*x_t;

        ui_t_all = sqrt(Omega_t_hat)*randn(1, n, 'gpuArray');

        S_tau_all = arrayfun(@simulateMC, ui_t_all, x_t, Omega_t_hat, t, T, M_t, A_t, B_t);%an array that stores S(tau) of each sample path starting at time t and state xt

        ri_all = gather(exp(-S_tau_all/lambda));  %(size: (1 X n))

        ui_t_all_arr = gather(ui_t_all); %concatenate ui_t_all_1 and ui_t_all_2 in an array

        Ehat_ru = (ui_t_all_arr*(ri_all')); %(size: (2 X 1))

        Ehat_r = sum(ri_all); %scalar

        u_t = Ehat_ru/Ehat_r;
 
        cost_dummy = cost_dummy + 0.5*u_t'*Omega_t/u_t; %cost(itr, i) = cost(itr, 1) + 0.5*u_t'*N_t*u_t, N_t = inv(Omega_t)  
        w_t = Omega_t*randn;
        x_t = A_t*x_t + B_t*u_t + w_t;
%         x_t = A_t*x_t + B_t*u_t;
    end

    cost_dummy = cost_dummy + 0.5*x_t'*M_t*x_t;
    cost(itr) = gather(cost_dummy);
end
    
figure(1)

hold on
set(gca, 'FontName', 'Arial', 'FontSize', 20)
xlabel('$\epsilon$', 'Interpreter','latex', 'FontSize', 30); ylabel('$J_0^*$', 'Interpreter','latex','FontSize', 30); 
set(gca,'LineWidth',1)
ax = gca;
ax.LineWidth = 1;
ax.Color = 'w';
plot(X_t, 216.3 - abs(cost_dummy), 'm')

%==========================================================================
clearvars -except X_t A_t B_t M_t Omega_t Omega_t_hat N_t lambda T

cost = zeros(length(X_t), 1);
for itr = 1:length(X_t)
    x_t = X_t(itr); 
    
    Phi_tp1 = M_t;
    kappa_tp1 = 0;
    
    for t=T-1:-1:0
        
        Phi_t = A_t'*Phi_tp1*A_t + M_t - A_t'*Phi_tp1*B_t/(B_t'*Phi_tp1*B_t + N_t)*B_t'*Phi_tp1*A_t;
        kappa_t = kappa_tp1 + (lambda/2)*log(det(Omega_t_hat)) + (lambda/2)*log(det(inv(Omega_t_hat) + (1/lambda)*B_t'*Phi_tp1*B_t));
        
        Phi_tp1 = Phi_t;
        kappa_tp1 = kappa_t;
        
    end 
    cost(itr) = kappa_t + 0.5*x_t'*Phi_t*x_t;
end
figure(1)

plot(X_t, cost, 'r')

%========================================================================
clearvars -except X_t A_t B_t M_t Omega_t Omega_t_hat N_t lambda T

cost = zeros(length(X_t), 1);

for itr = 1:length(X_t)
    x_t = X_t(itr);
    
    Theta_tp1 = M_t;
    kappa_tp1 = 0;
    
    for t=T-1:-1:0
        
        Theta_t = A_t'*Theta_tp1*A_t + M_t - A_t'*Theta_tp1*B_t/(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
        kappa_t = kappa_tp1 + 0.5*trace(Omega_t*Theta_tp1);
            
        Theta_tp1 = Theta_t;
        kappa_tp1 = kappa_t;
        
    end 
    cost(itr) = kappa_t + 0.5*x_t'*Theta_t*x_t;
end

plot(X_t, cost, 'b')

