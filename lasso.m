
clear all; close all; clc

data = csvread('sfo_data_clean.csv', 2);

[m, n] = size(data);

percent_train = 0.8;

shuffled_data = data(randperm(m),:);

x_train = shuffled_data(1:0.8*m,1:n-1);
y_train = shuffled_data(1:0.8*m,n);
x_test = shuffled_data(0.8*m:m, 1:n-1);
y_test = shuffled_data(0.8*m:m,n);

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

ridge_acc = sum(ridgePred == y_test)/length(y_test);
lasso_acc = sum(lassoPred == y_test)/length(y_test);

ridge_within_one = sum(abs(ridgePred - y_test) <= 1)/length(y_test);
lasso_within_one = sum(abs(lassoPred - y_test) <= 1)/length(y_test);

grouplassoOptions = struct('alpha', 1.0, 'multinomial', 'grouped');
grouplassoFit = cvglmnet(x_train, y_train, 'mgaussian', grouplassoOptions);
grouplassoPred = round(cvglmnetPredict(grouplassoFit, x_test, grouplassoFit.lambda_min));
cvglmnetPlot(grouplassoFit, -1);
grouplassoError = norm(y_test - grouplassoPred);
fprintf('Test error for group lasso model using least squares: %e\n', grouplassoError);

% figure(3)
% hold all
% plot(y_test(1:2:size(y_test,1)), 'o')
% plot(lassoPred(1:2:size(lassoPred,1)), 'ro');
% plot(ridgePred(1:2:size(ridgePred,1)), 'go');
figure(4)
plot(abs(lassoPred - y_test), 'ro');
title('Lasso Prediction Difference for Test Data');
xlabel('Data points');
ylabel('Difference from actual');
figure(5)
plot(abs(ridgePred - y_test), 'go');
title('Ridge Prediction Difference for Test Data');
xlabel('Data points');
ylabel('Difference from actual');
figure(6)
plot(abs(grouplassoPred - y_test), 'co');
title('Group Lasso Prediction Difference for Test Data');
xlabel('Data points');
ylabel('Difference from actual');
