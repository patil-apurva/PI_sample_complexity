function S_tau_all = simulateMC(u_t_prime, x_t_prime, Omega_t_hat, t, T, M_t, A_t, B_t)

    S_tau = 0; %the cost-to-go of the state dependent cost of a sample path
    
    for t_prime = t:1:T-1 % this loop is to compute S(tau_i)
                
        S_tau = S_tau + 0.5*(x_t_prime*M_t*x_t_prime); %add the state dependent running cost
        
        x_t_prime = A_t*x_t_prime + B_t*u_t_prime;

        u_t_prime = sqrt(Omega_t_hat)*randn; %u_t at new t_prime. Will be used in the next iteration 
    end

    S_tau = S_tau + 0.5*(x_t_prime*M_t*x_t_prime); %add the terminal cost to S_tau
   
    S_tau_all = S_tau;