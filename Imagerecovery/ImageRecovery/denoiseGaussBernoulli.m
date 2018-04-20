function [mout, vout] = denoiseGaussBernoulli(rhat, vr, pi0, mu0, sigvar0)
% This function is to implement signal denoising.
% rhat = x+w_d, x \sim Bernoulli Gaussian distribution with parameters pi0,
% mu0, sigvar0; w \sim N(0,vr).

% Input:
% - rhat: noisy signal (rhat = x+w_d)
% - vr: AWGN variance (Var(w_d) = vr)
% - pi0, mu0, sigvar0: signal prior parameters

% - EM_status\in{0,1,2}. 
% - denoiseGaussBernoulli, parameters known accurately
% - denoiseGaussBernoulli_EM1, all the parameters unknown and use EM learning
% without NNSPL method
% - denoiseGaussBernoulli_EM2, all the parameters are unknown use EM learning
% with NNSPL method

% Output:
% - mout: E(X | Rhat = rhat)
% - vout: Var(X | Rhat = rhat)

pi_t = pi0;
vx = sigvar0;
M = 0.5*log(vr./(vr+vx))+0.5*rhat.^2./vr-0.5*(rhat-mu0).^2./(vr+vx);
lambda = pi_t./(pi_t+(1-pi_t).*exp(-M));
m_t = (rhat.*vx+vr.*mu0)./(vr+vx);
V_t = vr.*vx./(vr+vx);

% Compute E{X|Rhat = rhat}
mout = lambda.*m_t;

% Compute Var{X|Rhat = rhat}
vout = lambda.*(m_t.^2+V_t)-mout.^2;
end