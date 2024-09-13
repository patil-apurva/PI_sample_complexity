LHS = zeros(5, 1);
RHS = zeros(5, 1);

eps = 10^-6;
eps_t2 = eps/T;
eps_t = sqrt(eps_t2);

x_t = x_0

n = 1;

for ii = 1: 100
    

    ui_t_all = sqrt(Omega_t_hat)*randn(1, n, 'gpuArray'); %u_t from reference policy

    S_tau_all = arrayfun(@simulateMC, ui_t_all, x_t, Omega_t_hat, 0, T, M_t, A_t, B_t); %an array that stores S(tau) of each sample path starting at time t and state xt

    ri_all = gather(exp(-S_tau_all/lambda)); 

    ui_t_all_arr = gather(ui_t_all); 

    Ehat_ru = (ui_t_all_arr*(ri_all'))/n; 

    Ehat_r = sum(ri_all)/n; 

    const1 = Ehat_r*sqrt(2*det(Omega_t_hat)*log(2*input_dim/beta_t));
    const2 = (eps_t*Ehat_r + norm(Ehat_ru))*sqrt(0.5*log(2/alpha_t));
    const3 = eps_t^2*Ehat_r^4;
    n_test = ((const1 + const2)^2)/const3;
    %         u_t = Ehat_ru/Ehat_r; %optimal u_t at the current time step
    %         w_t = sqrt(Omega_t)*randn;
    %         x_t = A_t*x_t + B_t*u_t + w_t;
    
    RHS(ii) = n_test;
    LHS(ii) = n;

    n = n+1;
end
        
