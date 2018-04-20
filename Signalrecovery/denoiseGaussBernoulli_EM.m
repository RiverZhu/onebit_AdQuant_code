function [mout, vout,  muout, piout, sigvarout] = denoiseGaussBernoulli_EM(rhat, vr, pi0, mu0, sigvar0, NNL)
% This function is to implement signal denoising and EM learning combined
% with NNSPL method.
% rhat = x+w_d, x \sim Bernoulli Gaussian distribution with parameters pi0,
% mu0, sigvar0; w \sim N(0,vr).

% Input:
% - rhat: noisy signal (rhat = x+w_d)
% - vr: AWGN variance (Var(w_d) = vr)
% - pi0, mu0, sigvar0: signal prior parameters
% NNL: the pre-generated NNSPL matrix used for NNSPL method

% Output:
% - mout: E(X | Rhat = rhat)
% - vout: Var(X | Rhat = rhat)
% - muout, piout, sigvarout: all the system parameters

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

% EM¡¡learning
piout = NNL*lambda;
muout = lambda'*m_t/sum(lambda);
sigvarout = lambda'*((mu0-m_t).^2+V_t)/sum(lambda);
end

