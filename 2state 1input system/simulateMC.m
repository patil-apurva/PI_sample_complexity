function S_tau_all = simulateMC(u_t_prime, x_t_prime_1,  x_t_prime_2, S_hat, t, T, M, A11, A12, A21, A22, B1, B2)

    S_tau = 0; %the cost-to-go of the state dependent cost of a sample path
    
    for t_prime = t:1:T-1 % this loop is to compute S(tau_i)
                
        S_tau = S_tau + 0.5*M*(x_t_prime_1*x_t_prime_1 + x_t_prime_2*x_t_prime_2); %add the state dependent running cost
        
        x_t_prime_1 = A11*x_t_prime_1 + A12*x_t_prime_2 + B1*u_t_prime;
        x_t_prime_2 = A21*x_t_prime_1 + A22*x_t_prime_2 + B2*u_t_prime;

        u_t_prime = S_hat*randn; %u_t_1 at new t_prime. Will be used in the next iteration 
    end

    S_tau = S_tau + 0.5*M*(x_t_prime_1*x_t_prime_1 + x_t_prime_2*x_t_prime_2); %add the terminal cost to S_tau
   
    S_tau_all = S_tau;