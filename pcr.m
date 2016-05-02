% PCR with 30 principal components gives roughly 62-63 percent accuracy,
% with 94 percent within one accuracy. Adding more principal components
% beyond this doesn't seem to help accuracy any further. Decreasing
% components will cause accuracy to suffer. Uses pca + ordinary regression.
% Plots loadings showing coefficients for each variable in generating the
% first 30 principal components.

% Looks like there are: 108 0s, 7 1s, 29 2s, 526 3s, 1507 4s, 740 5s, 40 6s
clear all; close all; clc

data = csvread('sfo_data_clean.csv', 2);

[m, n] = size(data);

percent_train = 0.9;

shuffled_data = data(randperm(m),:);

x_train = shuffled_data(1:0.8*m,1:n-1);
y_train = shuffled_data(1:0.8*m,n);
x_test = shuffled_data(0.8*m:m, 1:n-1);
y_test = shuffled_data(0.8*m:m,n);

[loadings, score, latent] = pca(x_train);
beta = regress(y_train - mean(y_train), score(:,1:30));
beta = loadings(:,1:30)*beta;
beta = [mean(y_train) - mean(x_train)*beta; beta];
y_pred = [ones(size(x_test,1),1) x_test]*beta;

y_pred = round(y_pred);
err = norm(y_pred - y_test);
acc = sum(y_pred == y_test)/length(y_test);
within_one = sum(abs(y_pred - y_test) <= 1)/length(y_test);

figure(1)
plot(1:80, loadings(:,1:10), '-');
xlabel('Var')
ylabel('Loading')
legend({'1st Component' '2nd Component' '3rd Component'  ...
	'4th Component' '5th Component' '6th Component'},'location','NE');

figure(2)
plot(1:80, loadings(:,7:12), '-');
xlabel('Var')
ylabel('Loading')
legend({'7th Component' '8th Component' '9th Component'  ...
	'10th Component' '11th Component' '12th Component'},'location','NE');

figure(3)
plot(1:80, loadings(:,13:18), '-');
xlabel('Var')
ylabel('Loading')
legend({'13th Component' '14th Component' '15th Component'  ...
	'16th Component' '17th Component' '18th Component'},'location','NE');

figure(4)
plot(1:80, loadings(:,19:24), '-');
xlabel('Var')
ylabel('Loading')
legend({'19th Component' '20th Component' '21st Component'  ...
	'22nd Component' '23rd Component' '24th Component'},'location','NE');

figure(5)
plot(1:80, loadings(:,25:30), '-');
xlabel('Var')
ylabel('Loading')
legend({'25th Component' '26th Component' '27th Component'  ...
	'28th Component' '29th Component' '30th Component'},'location','NE');