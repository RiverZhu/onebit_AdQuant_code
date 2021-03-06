function [xhat, vx] = GampEM2( A, y, tau, vx0, xhat0, pi0, mu0, sigvar0, Delta0, T, NNL_L, NNL_R)
% This function includes:
% (1) GAMP algorithm for one bit compressed sensing .
% (2) EM learning for mean and variance, and NNSPL method for non-zero
% probability.

% Input:
% - A: measurement matrix (m * n)
% - y: sign measurements (+1 or -1) (m * p)
% - tau: quantizer thresholds
% - vx0: initialization for the variance
% - xhat0: initialization for the signal
% - pi0, mu0, sigvar0, Delta0: initialized prior nonzero probability, prior mean, 
% prior variance, additive noise variance
% - T: number of iterations
% NNL_L, NNL_R: the pre-generated NNSPL matrix used for NNSPL method

% Output:
% - xhat: reconstructed signal (n * p)
% - vx: predicted MSE (n * p)



% Number of measurements and dimension of the signal
[m, n] = size(A);
p = size(y,2);
absdiff = @(x) sum(sum(abs(x)))/(size(x,1)*size(x,2));

% Initialize the estimates
xhat = xhat0;
vx = vx0;

% Initialize shat
shat = zeros(m, p);

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
    [xhat, vx,  mu0, pi0, sigvar0] = denoiseGaussBernoulli_EM2(rhat, vr, pi0, mu0, sigvar0, NNL_L, NNL_R);
    
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
