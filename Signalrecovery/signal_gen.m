%% Random structured sparse signal generation
% This file is to generate a random signal x and obtain the clean measurements z.

%% Signal generation
m =  n*bitpercompo; % number of measurements
m = floor(m);
Phi = (1/sqrt(m)) .* randn(m, n); % measurement matrix

% Set signal prior parameters
prior_pi = 0.25*ones(n,1);
prior_mean = 0.5;
prior_var = 1;

% Generate signal 
len = randi(25,4,1)+1;
len = len/sum(len)*25;
len = round(len);
if sum(len)~=25
    [~,index] = max(len);
    len(index) = len(index)+25-sum(len);
end
j1 = len(1);
j2 = len(2);
j3 = len(3);
j4 = len(4);

label = randi(100,1,4);
label = sort(label);
i1 = label(1);
i2 = label(2);
i3 = label(3);
i4 = label(4);
while(((i2-i1)<j1)||((i3-i2)<j2)||((i4-i3)<j3)||((n+1-i4)<j4))
    label = randi(100,1,4);
    label = sort(label);
    i1 = label(1);
    i2 = label(2);
    i3 = label(3);
    i4 = label(4);
end
x = zeros(n,1);
x(i1:(i1+j1-1)) = prior_mean*ones(j1,1)+prior_var*randn(j1,1);
x(i2:(i2+j2-1)) = prior_mean*ones(j2,1)+prior_var*randn(j2,1);
x(i3:(i3+j3-1)) = prior_mean*ones(j3,1)+prior_var*randn(j3,1);
x(i4:(i4+j4-1)) = prior_mean*ones(j4,1)+prior_var*randn(j4,1);

%% Obtain clean measurements
z = Phi*x; % the clean measurements
% SNR = 20;
% v = (norm(z))^2/10^(SNR/10)/m;
v = 0;  % noiseless
noise = sqrt(v) * randn(m, 1);