%% Codes for image recovery
% written by Hangting Cao and Jiang Zhu, April 18th, 2018.
% This file implements the proposed GAMP-EM-AD_NNSPL algorithm to recover
% the image 'Cameraman' while compared with GAMP-oracle-AD and GAMP-EM-AD.

%% Initialization
clc; clear; close all;
T = 500; % maximum number of iterations in GAMP
imageread % the file imageread.m reads the image and obtains the clean 
          % measurements z

% Initialization of GAMP
vx0 = 1 * ones(size(x));
xhat0 = zeros(size(x));
pi_0 = 0.5 * ones(size(x));
pr_mean_0 = 0;
prior_var_0 = 1;
Delta0 = v;

% Define the NNSPL matrix
NNL_L = zeros(n,n);
NNL_L(1,2) = 1; NNL_L(n,n-1) = 1;
for i = 1:(n-2)
    NNL_L(i+1,i:i+2) = [1 0 1];
end
NNL_R = zeros(p,p);
NNL_R(1,2) = 1; NNL_R(p,p-1) = 1;
for i = 1:(p-2)
    NNL_R(i+1,i:i+2) = [1 0 1];
end
NNL_R = NNL_R';

%% Define the blocksize for AD
global blockSize
blockSize = 15;

%% Reconstruction

% GAMP-EM-AD-NNSPL
[xhat1, vx1] = adaptiveGAMP_NNL( Phi, z, vx0, xhat0, pi_0, pr_mean_0,...
prior_var_0, Delta0, T, noise, NNL_L, NNL_R); % implement GAMP-EM-AD-NNSPL
xhat1 = xhat1*x_max; % recover x from the scaling
[xhat1,L] = midwt(xhat1,h,L); 
PSNR1 = psnr(uint8(xhat1),uint8(x_ref)); % calculate the psnr

% GAMP-EM-AD
[xhat2, vx2] = adaptiveGAMP( Phi, z, vx0, xhat0, pi_0, pr_mean_0,prior_var_0,...
Delta0, T, noise);
xhat2 = xhat2*x_max;
[xhat2,L] = midwt(xhat2,h,L);
PSNR4 = psnr(uint8(xhat2),uint8(x_ref)); 

% GAMP-oracle-AD
[xhat3, vx3] = adaptiveGamporacle( Phi, z, vx0, xhat0, prior_pi, prior_mean,...
prior_var, Delta0, T, noise);
xhat3 = xhat3*x_max;
[xhat3,L] = midwt(xhat3,h,L);
PSNR6 = psnr(uint8(xhat3),uint8(x_ref)); 

%% Plot
figure(1)
subplot(2,2,1)
imshow(uint8(x_ref)); title(sprintf('original image'),'FontSize',10)
subplot(2,2,2)
imshow(uint8(xhat3)); title(sprintf('GAMP-oracle-AD,PSNR = %0.2f dB',PSNR6),'FontSize',10)
subplot(2,2,3)
imshow(uint8(xhat2)); title(sprintf('GAMP-EM-AD,PSNR = %0.2f dB',PSNR4),'FontSize',10)
subplot(2,2,4)
imshow(uint8(xhat1)); title(sprintf('GAMP-EM-AD-NNSPL,PSNR = %0.2f dB',PSNR1),'FontSize',10)