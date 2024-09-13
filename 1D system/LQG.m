clearvars
close all

A_t = 0.85;
B_t = 0.10;
M_t = 3.0; %state cost wt
Omega_t = 1.5; %noise covariance
Omega_t_hat = Omega_t/(B_t^2);
N_t = inv(Omega_t); %control cost wt
lambda = N_t/inv(Omega_t_hat);

T = 30; %time horizon
x_0 = -3;

Theta_tp1 = M_t;
kappa_tp1 = 0;
k_tp1 = 0;
    
    for t=T-1:-1:0

        Theta_t = A_t'*Theta_tp1*A_t + M_t - A_t'*Theta_tp1*B_t/(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
        kappa_t = kappa_tp1 + 0.5*trace(Omega_t*Theta_tp1);
        k_t = k_tp1 + lambda/2*log(det(Omega_t_hat)) + lambda/2*log(det(inv(Omega_t_hat) + 1/lambda*B_t'*Theta_tp1*B_t));
         
        Theta_tp1 = Theta_t;
        kappa_tp1 = kappa_t;
        k_tp1 = k_t;
        
        K_t_LQG = -inv(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
        
        H_t_hat = B_t'*Theta_tp1*B_t + N_t;
        G_t_hat = A_t'*Theta_tp1*B_t;
        K_t_PI = -H_t_hat\G_t_hat';
        
        
        
    end
  
   J_LQG = kappa_t + 0.5*x_0'*Theta_t*x_0
   J_PI = k_t + 0.5*x_0'*Theta_t*x_0
   
   
   
   
   