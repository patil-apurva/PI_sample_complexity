clearvars
close all
rng('default')
seed = 1234;
rng( seed ); % Reset the CPU random number generator.

state_dim = 2; %number of states
input_dim = 1; %number of inputs

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

n = 10^3;

alpha_t = 0.01;
beta_t = 0.0001;

T = 30; %time horizon

x_0 = [-3; -3]; %initial state

Theta_t = M_t; %terminal Theta, parameter for Riccati equation
kappa_t = 0; %terminal kappa, parameter for Riccati equation
K_t_all = cell(T,1);

kappa_t_all(T+1) = {kappa_t};
Theta_t_all(T+1) = {Theta_t};

for t=T-1:-1:0
    Theta_tp1 = Theta_t;
    kappa_tp1 = kappa_t;
    
    K_t = -inv(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
    Theta_t = A_t'*Theta_tp1*A_t + M_t - A_t'*Theta_tp1*B_t/(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
    
    kappa_t = kappa_tp1 + 0.5*trace(Omega_t*Theta_tp1);
    
    K_t_all(t+1) = {K_t};
    kappa_t_all(t+1) = {kappa_t};
    Theta_t_all(t+1) = {Theta_t};
end

x_t = x_0;
u_t_all_LQG = [];
x_t_all_LQG = [x_t];
J_t_all_LQG = [];

for t=0:T-1
    
    kappa_t = cell2mat(kappa_t_all(t+1));
    Theta_t = cell2mat(Theta_t_all(t+1));
    J_t = kappa_t + 0.5*(x_t')*Theta_t*x_t;
    J_t_all_LQG = [J_t_all_LQG, J_t];
    
    K_t = cell2mat(K_t_all(t+1));
    u_t = K_t*x_t;
    u_t_all_LQG = [u_t_all_LQG, u_t];
    
    x_t = A_t*x_t + B_t*u_t;
    x_t_all_LQG = [x_t_all_LQG, x_t];
  
end

kappa_t = cell2mat(kappa_t_all(T+1));
Theta_t = cell2mat(Theta_t_all(T+1));
J_t = kappa_t + 0.5*(x_t')*Theta_t*x_t;
J_t_all_LQG = [J_t_all_LQG, J_t];

x_t = x_0;

u_t_all = [];
x_t_all = [x_0];
eps_t_all = [];

for t = 0:T-1
     
    ui_t_all = S_hat*randn(1, n, 'gpuArray');

    S_tau_all = arrayfun(@simulateMC, ui_t_all, x_t(1), x_t(2), S_hat, t, T, M, A_t(1,1), A_t(1,2), A_t(2,1), A_t(2,2), B_t(1), B_t(2));%an array that stores S(tau) of each sample path starting at time t and state xt

    ri_all = gather(exp(-S_tau_all/lambda)); %(size: (1 X n))

    ui_t_all_arr = gather(ui_t_all); %array of ui_t_all

    Ehat_ru = (ui_t_all_arr*(ri_all'))/n; %(size: (1 X 1))

    Ehat_r = sum(ri_all)/n; %scalar
     
    u_t = Ehat_ru/Ehat_r;
     
        if(any(isnan(u_t(:))))
            t
            fprintf("error!")
            return
        end
        
        const1 = Ehat_r*sqrt(2*norm(Omega_t_hat)/n * log(2*input_dim/beta_t));
        const2 = norm(Ehat_ru, Inf)*sqrt(1/(2*n)*log(2/alpha_t));
        const3 = (Ehat_r - sqrt(1/(2*n)*log(2/alpha_t)))*Ehat_r;
    
        eps_t = (const1 + const2)/const3;
        eps_t_all = [eps_t_all, eps_t];
        
        u_t_all = [u_t_all, u_t];
       
        x_t = A_t*x_t + B_t*u_t;
    
        x_t_all = [x_t_all, x_t];
end
    
    figure(5)
    hold on
    set(gca, 'FontName', 'Arial', 'FontSize', 30)
    xlabel('$t$', 'Interpreter','latex', 'FontSize', 40); ylabel('$x_1, x_2$', 'Interpreter','latex','FontSize', 40); 
    set(gca,'LineWidth',1)
    ax = gca;
    ax.LineWidth = 1;
    ax.Color = 'w';
    % axis equal;
    xlim([0, T]);
%     ylim([-1, 3]);
    plot(linspace(0, T, T+1),  x_t_all(1,:), 'b', 'Linewidth', 1);
    plot(linspace(0, T, T+1), x_t_all_LQG(1,:), '--r', 'Linewidth', 2);
    plot(linspace(0, T, T+1),  x_t_all(2,:), 'b', 'Linewidth', 1);
    plot(linspace(0, T, T+1), x_t_all_LQG(2,:), '--r', 'Linewidth', 2);
    lgd = legend('PI', 'Analytical', 'FontSize', 30);
    

    figure(6)
    hold on
    set(gca, 'FontName', 'Arial', 'FontSize', 30)
%     xlabel('$t$', 'Interpreter','latex', 'FontSize', 40); ylabel('$u$', 'Interpreter','latex','FontSize', 40); 
    set(gca,'LineWidth',1)
    ax = gca;
    ax.LineWidth = 1;
    ax.Color = 'w';
    % axis equal;
    xlim([0, T]);
    ylim([-0.1, 0.15]);
    plot(linspace(0, T-1, T), u_t_all_LQG + eps_t_all, 'k', 'Linewidth', 2);
    h1 = plot(linspace(0, T-1, T), u_t_all_LQG - eps_t_all, 'k', 'Linewidth', 2);
   
    x2 = [linspace(0, T-1, T), fliplr(linspace(0, T-1, T))];
    inBetween = [u_t_all_LQG + eps_t_all, fliplr(u_t_all_LQG - eps_t_all)];
    fill(x2, inBetween, 'y');
    
%     plot(linspace(0, T-1, T), eps_t_all, 'k', 'Linewidth', 2);

    h2 = plot(linspace(0, T-1, T),  u_t_all(1,:), 'b', 'Linewidth', 1);
    h3 = plot(linspace(0, T-1, T), u_t_all_LQG(1,:), '--r', 'Linewidth', 2);
    
%     legend([h1, h2, h3], {'$u_t\pm \epsilon_t$', 'PI', 'Analytical'},'Interpreter','latex', 'FontSize', 30)
    
    
