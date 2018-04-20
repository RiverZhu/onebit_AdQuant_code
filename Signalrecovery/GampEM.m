function [xhat, vx] = GampEM( A, y, tau, init, pi0, mu0, sigvar0, Delta0, T, NNL)
% This function includes:
% (1) GAMP algorithm for one bit compressed sensing .
% (2) EM learning for mean and variance, and NNSPL method for non-zero
% probability.

% Input:
% - A: measurement matrix (m x n)
% - y: sign measurements (+1 or -1) (m x 1)
% - tau: quantizer thresholds
% - init: initialization for the signal and variance [xhat0, vx0]
% - pi0, mu0, sigvar0, Delta0: initialized prior nonzero probability, prior mean, prior
% variance, additive noise variance
% - T: number of iterations
% NNL: the pre-generated NNSPL matrix used for NNSPL method

% Output:
% - xhat: reconstructed signal (n x 1)
% - vx: predicted MSE (n x 1)

% Number of measurements and dimension of the signal
[m, n] = size(A);
absdiff = @(x) sum(abs(x))/length(x);

% Initialize the estimates
if(numel(init) == 2)
    xhat = init(1)*zeros(n, 1);
    vx = init(2)*ones(n, 1);
else
    xhat = init(:, 1);
    vx = init(:, 2);
end

% Initialize shat
shat = zeros(m, 1);

% Set damping factor
dampFac = 0.5;

% Previous estimate
xhatprev = xhat;
shatprev = shat;
vxprev = vx;

% Number of unchanged iterations
count = 3;

% Hadamard product of the matrix
AA = A.*A;

% Perform estimation
for t = 1:T
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Measurement update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear
    vp = AA*vx;
    phat = A*xhat - vp.*shat;
    
    % Truncated Gaussian
    [ez, vz] = GaussianMomentsComputation(y, tau, phat, vp, Delta0);
 
    % Non-Linear
    shat = (ez-phat)./vp;
    vs = (1-vz./vp)./vp;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Estimation update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear
    vr = 1./(AA' * vs);
    rhat = xhat + vr .* (A' * shat);
    
    % Non-linear and EM learning, noise variance is known 
    [xhat, vx,  mu0, pi0, sigvar0] = denoiseGaussBernoulli_EM(rhat, vr, pi0, mu0, sigvar0, NNL);
    
    %Damp
    xhat = dampFac*xhat + (1-dampFac)*xhatprev;
    shat = dampFac*shat + (1-dampFac)*shatprev;
    vx = dampFac*vx + (1-dampFac)*vxprev;
    
    % If without a change
    if(absdiff(xhat - xhatprev) < 1e-6)
        count = count - 1;
    end
    
    % Save previous xhat
    xhatprev = xhat;
    shatprev = shat;
    vxprev = vx;
    
    % Stopping criterion
    if(count <= 0)
        break;
    end
end

end
