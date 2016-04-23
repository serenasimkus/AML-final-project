
clear all; close all; clc

data = csvread('sfo_data_clean.csv', 2);

[m, n] = size(data);

percent_train = 0.8;

x_train = data(1:0.8*m,1:n-1);
y_train = data(1:0.8*m,n);
x_test = data(0.8*m:m, 1:n-1);
y_test = data(0.8*m:m,n);

lassoOptions = struct('alpha', 1.0);
lassoFit = cvglmnet(x_train, y_train, 'gaussian', lassoOptions);
lassoPred = round(cvglmnetPredict(lassoFit, x_test, lassoFit.lambda_min));
cvglmnetPlot(lassoFit, -1);
lassoError = norm(y_test - lassoPred);
fprintf('Test error for lasso model using least squares: %e\n', lassoError);

ridgeOptions = struct('alpha', 0.0);
ridgeFit = cvglmnet(x_train, y_train, 'gaussian', ridgeOptions);
ridgePred = round(cvglmnetPredict(ridgeFit, x_test, ridgeFit.lambda_min));
cvglmnetPlot(ridgeFit, -1);
ridgeError = norm(y_test - ridgePred);
fprintf('Test error for ridge regression model using least squares: %e\n', ridgeError);

% figure(3)
% hold all
% plot(y_test(1:2:size(y_test,1)), 'o')
% plot(lassoPred(1:2:size(lassoPred,1)), 'ro');
% plot(ridgePred(1:2:size(ridgePred,1)), 'go');
figure(4)
plot(abs(lassoPred - y_test), 'ro');
figure(5)
plot(abs(ridgePred - y_test), 'go');
