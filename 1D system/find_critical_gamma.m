state_dim = 1;
input_dim = 1;
A_t = 0.85;
B_t = 0.10;
M_t = 3.0; %state cost wt
Omega_t = 1.5; %noise covariance
Omega_t_hat = Omega_t/(B_t^2);
N_t = inv(Omega_t); %control cost wt

Theta_t = M_t; %terminal Theta
Phi_t = M_t; %terminal Phi
gamma = 0.4;
T = 30; %time horizon

for t=T-1:-1:0
    Theta_tp1 = Theta_t;
    K_t = -inv(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t;
    
    Atilde_t = A_t + B_t*K_t;
    Mtilde_t = M_t + K_t'*N_t*K_t;
    Ntilde_t = K_t'*N_t;
    Theta_t = A_t'*Theta_tp1*A_t + M_t - A_t'*Theta_tp1*B_t/(B_t'*Theta_tp1*B_t + N_t)*B_t'*Theta_tp1*A_t; 

    Phi_tp1 = Phi_t;
    S_t = gamma*eye(input_dim) - N_t; 
    F_t = -inv(B_t'*Phi_tp1*B_t - S_t)*(Ntilde_t' + B_t'*Phi_tp1*Atilde_t);

    test_mat = N_t + B_t'*Phi_tp1*B_t - 2*gamma*eye(input_dim);  
    
            
    d = eig(test_mat);
    if(any(d>=0))
        disp('Test matrix is not negative definite')
        t
        test_mat

        break
    end
           
    Phi_t = Mtilde_t + Atilde_t'*Phi_tp1* Atilde_t + (Ntilde_t + Atilde_t*Phi_tp1*B_t)*F_t; 
end


