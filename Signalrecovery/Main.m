%% Codes for synthetic signal recovery
% written by Jiang Zhu and Hangting Cao, April 18th, 2018.
% This file implements the proposed GAMP-EM-AD_NNSPL algorithm to recover
% the synthetic signal recovery.

%% Initialization
clc; clear; close all;

%% Define the blocksize of AD
global blockSize
blockSize = 20;

%% Parameter initialization
n = 100; % the dimension of signal x
T = 100; % the maximum number of GAMP iterations
bitpercompo = 3; % the rate(bits/signal entry)
computeSnr = @(sig, noise) 10*log10((norm(sig)^2)/(norm(noise)^2)); % function calculating the reconstruction SNR
signal_gen % the file signal_gen.m generates a random signal x and obtains the clean measurements z

% Define the NNSPL matrix
NNL = zeros(n,n);
NNL(1,2) = 1; NNL(n,n-1) = 1;
for i = 1:(n-2)
    NNL(i+1,i:i+2) = [1/2 0 1/2];
end

%% Implement GAMP-EM-AD-NNSPL
% Initialization for GAMP
vx0 = 10 * ones(size(x));
xhat0 = zeros(size(x));
init0 = [xhat0, vx0];
pi_0 = 0.5;
pr_mean_0 = 0;
prior_var_0 = 10;
Delta0 = v;

% Perform algorithm
[xhat, vx] = adaptiveGAMP_NNL( Phi, z, init0, pi_0, pr_mean_0, prior_var_0, Delta0, T, noise, NNL);

% Calculate the reconstruction SNR
recon_snr = computeSnr(x, x-xhat);

%% Plot
figure(1)
stem(xhat)
hold on;
stem(x)
legend('xhat', 'x');



