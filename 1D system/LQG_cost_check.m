x_0 = -3; %initial state
A_t = 0.85;
B_t = 0.10;
M_t = 3.0; %state cost wt
Omega_t = 1.5; %noise covariance
N_t = inv(Omega_t); %control cost wt
T = 30;
traj_num = 100;

eps_vec = [10^-7, 10^-6];

%LQG cost
Theta_tp1 = M_t;
kappa_tp1 = 0;
K_all = zeros(T-1,1);

for t = T-1:-1:0
    
    K_t = -inv(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;

    K_all(t+1) = K_t;

    Theta_t = A_t'*Theta_tp1*A_t + M_t - A_t'*Theta_tp1*B_t/(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
    kappa_t = kappa_tp1 + 0.5*trace(Omega_t*Theta_tp1);
    
    Theta_tp1 = Theta_t;
    kappa_tp1 = kappa_t;
end    

true_cost_lqg = kappa_t + 0.5*x_0'*Theta_t*x_0

costs = zeros(traj_num, 1);
for ii = 1:exp_num
    for j = 1:traj_num
        cost = 0;
        x_t = x_0; %initial state

        for t = 0:T-1
            u_t = K_all(t+1)*x_t;
            cost = cost + 0.5*(x_t')*M_t*x_t + 0.5*(u_t')*N_t*u_t; %cost at the current time step and the current trajectory 

            w_t = sqrt(Omega_t)*randn;
            x_t = A_t*x_t + B_t*u_t + w_t;
        end

        cost = cost + 0.5*(x_t')*M_t*x_t;
        costs(j) = cost;    
    end
        scatter(log10(eps), mean(costs), 100, 'x', 'r')
end