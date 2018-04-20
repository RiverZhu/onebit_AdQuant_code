function [ xhat, vx ] = adaptiveGAMP_NNL( A, z, vx0, xhat0, pi0, mu0, sigvar0, Delta0, T, noise, NNL_L, NNL_R)
% This function is to implement the GAMP-EM-AD-NNSPL.

% Input:
% - A: measurement matrix (m * n)
% - z: clean measurements z = A*x (m * p)
% - vx0: initialization for the variance
% - xhat0: initialization for the signal
% - pi0, mu0, sigvar0, Delta0: initialized prior nonzero probability, prior mean, 
% prior variance, additive noise variance
% - T: number of iterations
% - noise, NNL_L, NNL_R: the pre-generated noise, NNSPL matrix used for NNSPL method
%
% Output:
% - xhat: reconstructed signal (n * p)
% - vx: predicted MSE (n * p)

% Measurements block size
global blockSize

% Problem dimensions
[m, n] = size(A);
p = size(z,2);

% Initial reconstruction
if(m >= n)
    minit = n;
else
    minit = m;
end

% Initial thresholds
tauinit = zeros(minit, p);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Quantized measurements
y = sign(z(1:minit,:)+noise(1:minit,:)+tauinit);

[xhat, vx] = GampEM2( A(1:minit, :), y, tauinit, vx0, xhat0, pi0, mu0, sigvar0, Delta0, T, NNL_L, NNL_R);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adaptively acquire
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Current measurement
mlow = minit + 1;

% Highest measurement in the block
mhigh = mlow + blockSize - 1;

% Initial tau
tau = tauinit;

while(mhigh <= m)
    
     % Thresholds
    tau = [tau; (-A(mlow:mhigh, :)*xhat)];
    
    % Measurements
    y = sign(z(1:mhigh,:) + noise(1:mhigh,:)+ tau);
    
    % Initialization
    xhat0 = xhat;
    vx0 = vx;

    % Reconstruct
    [xhat, vx] = GampEM2( A(1:mhigh, :), y, tau, vx0, xhat0, pi0, mu0, sigvar0, Delta0, T, NNL_L, NNL_R);
    
    % Update next block indices
    mlow = mhigh + 1;
    mhigh = mlow + blockSize - 1;
    
end

if(mlow <= m)
    % Thresholds
    tau = [tau; (-A(mlow:m, :)*xhat)];
    
    % Measurements
    y = sign(z + noise + tau);
    
    % Initialization
    xhat0 = xhat;
    vx0 = vx;

    % Reconstruct
    [xhat, vx] = GampEM2( A, y, tau, vx0, xhat0, pi0, mu0, sigvar0, Delta0, T, NNL_L, NNL_R);
end

end

