function [ xhat, vx ] = adaptiveGAMP_NNL( A, z, init, pi0, mu0, sigvar0, Delta0, T, noise,NNL)
% This function is to implement the GAMP-EM-AD-NNSPL.

% Input:
% - A: measurement matrix (m * n)
% - z: clean measurements z = A*x (m * 1)
% - init: initialization for the signal and variance [xhat0, vx0]
% - pi0, mu0, sigvar0, Delta0: initialized prior nonzero probability, prior mean, 
% prior variance, additive noise variance
% - T: number of iterations
% - noise, NNL: the pre-generated noise, NNSPL matrix used for NNSPL method

% Output:
% - xhat: reconstructed signal (n * 1)
% - vx: predicted MSE (n * 1)

% Problem dimensions
[m, n] = size(A);

% Initial reconstruction
if(m > n)
    minit = n;
else
    minit = m/2;
end

% Initial thresholds
tauinit = zeros(minit,1);

% Measurements block size
global blockSize
% blockSize = 200;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Quantized measurements
y = sign(z(1:minit)+noise(1:minit)+tauinit);

[xhat, vx] = GampEM( A(1:minit, :), y, tauinit, init, pi0, mu0, sigvar0, Delta0, T, NNL);

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
    y = sign(z(1:mhigh) + noise(1:mhigh)+ tau);
    
    % Initialization
    init = [xhat, vx];

    % Reconstruct
    [xhat, vx] = GampEM( A(1:mhigh, :), y, tau, init, pi0, mu0, sigvar0, Delta0, T, NNL);
    
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
    init = [xhat, vx];

    % Reconstruct
    [xhat, vx] = GampEM( A, y, tau, init, pi0, mu0, sigvar0, Delta0, T, NNL);
    
end

end

