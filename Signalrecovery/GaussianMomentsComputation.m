function [mout, vout] = GaussianMomentsComputation(y, tauin, phatin, vpin, vnin)
% This function is to returns posterior mean and variance for E(z|y)
% and Var(z|y), where y=sign(z+tau+w)

% Input:
% - y: sign measurements
% - tauin: thresholds
% - phatin: prior mean of z
% - vpin:  prior variance of z
% - vnin: additive noise of w

% Output:
% - mout: E(Z | Y = y)
% - vout: Var(Z | Y = y)

% E(Z | Y = y)
alpha = (phatin+tauin) ./ sqrt(vpin+vnin);
C = y.*alpha;
CDF = normcdf(C,0,1);
ll = log(CDF);

% Find bad values that cause log cdf to go to infinity
I = find(C < -30);              
ll(I) = -log(2)-0.5*C(I).^2+log(erfcx(-C(I)/sqrt(2))); % This expression is equivalent to log(normpdf(C)) for all negative arguments 
                                                       % greater than -30 and is numerically robust for all values smaller.  
                                                       
temp = exp(log(normpdf(alpha))-ll)./sqrt(vpin+vnin);
mout = phatin+sign(y).*vpin.*temp;

% Var(Z | Y = y)
temp1 = (phatin.*vnin-tauin.*vpin)./(vpin+vnin)-phatin;
vout = vpin+sign(y).*temp.*temp1.*vpin-(vpin.*temp).^2;
end

