%% Image read
% This file is to read the image and generate the clean measurements z.

%% read the image
filename= 'cameraman.jpg';
Image = imread(filename);
Image = rgb2gray(Image);
x = imresize(Image,0.5); % change the image's size to 128 * 128
x = double(x);
h = daubcqf(4);
[x,L] = mdwt(x, h); % implement Haar-wavelet transform
[n,p] = size(x);
x_max = max(max(abs(x)));
x = x/x_max; % implement the scaling
thresh = 4e-4;
sparsity = sum(sum(abs(x)>thresh))/n/p; 
x = x.*(abs(x)>thresh); % set the small coefficients to 0
[x_ref,L] = midwt(x,h,L);
x_ref = x_ref * x_max;

%% Generate observations
m = 3*n; % the number of measurements

% Calculations of some priors
x_vec = x(:);
x_vec = x_vec(x_vec~=0);
prior_pi = sparsity*ones(n,p);
prior_mean = mean(x_vec);
prior_var = var(x_vec);

% generation of observation y
Phi = (1/sqrt(m)) .* randn(m, n); % measurement matrix
Phi = Phi ./ (sqrt(sum(Phi.*Phi))); % normalize the measurement matrix
z = Phi * x;
v = 0;  % variance of noise
noise = sqrt(v) * randn(m, p);