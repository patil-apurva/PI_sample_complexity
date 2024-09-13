clearvars
close all
rng('default') 

state_dim = 1; %number of states
input_dim = 1; %number of inputs
A_t = 0.85;
B_t = 0.10;
M_t = 3.0; %state cost wt
Omega_t = 1.5; %noise covariance
Omega_t_hat = Omega_t/(B_t^2);
N_t = inv(Omega_t); %control cost wt

T = 30; %time horizon
alpha_t = 0.0001;
beta_t = 0.0001;

x_0 = -3; %initial state

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

lambda = N_t/inv(Omega_t_hat);
x_t = x_0;

u_t_all = [];
x_t_all = [x_0];
eps_t_all = [];
n = 10^3;

for t = 0:T-1
     
    ui_t_all = sqrt(Omega_t_hat)*randn(1, n, 'gpuArray');

    S_tau_all = arrayfun(@simulateMC, ui_t_all, x_t, Omega_t_hat, t, T, M_t, A_t, B_t);%an array that stores S(tau) of each sample path starting at time t and state xt

    ri_all = gather(exp(-S_tau_all/lambda)); %(size: (1 X n))

    ui_t_all_arr = gather(ui_t_all); %concatenate ui_t_all_1 and ui_t_all_2 in an array

    Ehat_ru = (ui_t_all_arr*(ri_all'))/n; %(size: (1 X 1))

    Ehat_r = sum(ri_all)/n; %scalar
     
    u_t = Ehat_ru/Ehat_r;
     
    if(any(isnan(u_t(:))))
        t
        fprintf("error!")
        return
    end
    
    const1 = Ehat_r*sqrt(2*norm(Omega_t_hat)/n * log(2*input_dim/beta_t));
    const2 = norm(Ehat_ru)*sqrt(1/(2*n)*log(2/alpha_t));
    
    sum_ECs = 0;
    
    for s = t:T
        G_s = [];
        Omega_bar_s = [];
        
        for s_prime = t:s-1
            G_s = [G_s, A_t^(s-1-s_prime)];
            Omega_bar_s = blkdiag (Omega_bar_s, Omega_t);
        end
        
        ECs = 0.5*trace(M_t*(G_s*Omega_bar_s*G_s.' + A_t^(s-t)*(x_t*x_t.')*(A_t^(s-t)).'));
        sum_ECs = sum_ECs + ECs;
    end
    
    const3 = exp(sum_ECs/lambda);
    
    eps_t = (const1 + const2)*(const3/Ehat_r);
    
    eps_t_all = [eps_t_all, eps_t];
    
    u_t_all = [u_t_all, u_t];

    x_t = A_t*x_t + B_t*u_t;

    x_t_all = [x_t_all, x_t];
end
    
    figure(3)
    hold on
    set(gca, 'FontName', 'Arial', 'FontSize', 30)
    xlabel('$t$', 'Interpreter','latex', 'FontSize', 40); ylabel('$x$', 'Interpreter','latex','FontSize', 40); 
    set(gca,'LineWidth',1)
    ax = gca;
    ax.LineWidth = 1;
    ax.Color = 'w';
    % axis equal;
    xlim([0, T]);
%     ylim([-1, 3]);
    plot(linspace(0, T, T+1),  x_t_all, 'b', 'Linewidth', 1);
    plot(linspace(0, T, T+1), x_t_all_LQG, '--r', 'Linewidth', 2);
    lgd = legend('PI', 'Analytical');
    

    figure(4)
    hold on
    set(gca, 'FontName', 'Arial', 'FontSize', 30)
    xlabel('$t$', 'Interpreter','latex', 'FontSize', 40); ylabel('$u$', 'Interpreter','latex','FontSize', 40); 
    set(gca,'LineWidth',1)
    ax = gca;
    ax.LineWidth = 1;
    ax.Color = 'w';
    % axis equal;
    xlim([0, T]);
    ylim([-1, 3]);
    plot(linspace(0, T-1, T), u_t_all_LQG + eps_t_all, 'k', 'Linewidth', 2);
    plot(linspace(0, T-1, T), u_t_all_LQG - eps_t_all, 'k', 'Linewidth', 2);
    
    x2 = [linspace(0, T-1, T), fliplr(linspace(0, T-1, T))];
inBetween = [u_t_all_LQG + eps_t_all, fliplr(u_t_all_LQG - eps_t_all)];
fill(x2, inBetween, 'y')

    plot(linspace(0, T-1, T),  u_t_all, 'b', 'Linewidth', 1);
    plot(linspace(0, T-1, T), u_t_all_LQG, '--r', 'Linewidth', 2);
   
    legend('u+eps', 'u-eps', 'error', 'PI', 'Analytical', 'Interpreter','latex')
    
    title('New Bound')
