clc 
close all
clear variables

%date: 6th Feb 2021
%VM
font = 14;
linewidth = 2;
format long;
%addpath('/Users/varunmannam/Desktop/Spring21/Research_S21/Feb21/SPIE_SR/old_SR_files/mse_plots/');
addpath('/Users/varunmannam/Desktop/Spring21/Research_S21/Feb21/0702/SR_new_results/mse_loss/'); %new simulations

lr1 = load('DenseED_train_RMSE_0702_2021_20210207_000054_vm41.txt'); %new result with VM4 SR result MSE 
lr2 = load('DenseED_test_RMSE_0702_2021_20210207_000054_vm41.txt'); %new result with VM52 SR result MSE 

epochs = 200;
x=[1:epochs]';

figure(1), 
yyaxis left
plot(x,lr1,'b', 'Linewidth', linewidth);
ylabel('MSE loss');
ylim([0.4 5.5]);
%set(gca, 'YScale', 'log')
yyaxis right
plot(x,lr2,'r', 'Linewidth', linewidth);
%ylim([0.05 0.12]);
xlabel('Epochs');
ylabel('MSE loss');
title('Configuration: (9,18,9) with 400 images');
legend('Train MSE ', 'Test MSE', 'Location', 'best');
%colormap(gray); colorbar;
%set(gca, 'YScale', 'log')
set(gca,'FontSize',font);


lr3 = load('DenseED_train_RMSE_0702_2021_20210207_000620_vm521.txt'); %new result with VM4 SR result MSE 
lr4 = load('DenseED_test_RMSE_0702_2021_20210207_000620_vm521.txt'); %new result with VM52 SR result MSE 

figure(2), 
yyaxis left
plot(x,lr3,'b', 'Linewidth', linewidth);
ylabel('MSE loss');
ylim([0.4 5.5]);
yyaxis right
plot(x,lr4,'r', 'Linewidth', linewidth);
%ylim([0.2 0.5]);
xlabel('Epochs');
ylabel('MSE loss');
title('Configuration: (8,8,8) with 400 images');
legend('Train MSE ', 'Test MSE', 'Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);

lr5 = load('DenseED_train_RMSE_0702_2021_20210207_014432_vm6141.txt'); %new result with VM4 SR result MSE 
lr6 = load('DenseED_test_RMSE_0702_2021_20210207_014432_vm6141.txt'); %new result with VM52 SR result MSE 

figure(3), 
yyaxis left
plot(x,lr5,'b', 'Linewidth', linewidth);
ylabel('MSE loss');
ylim([0.4 5.5]);
yyaxis right
plot(x,lr6,'r', 'Linewidth', linewidth);
%ylim([0.05 0.12]);
xlabel('Epochs');
ylabel('MSE loss');
title('Configuration: (9,18,9) with 750 images');
legend('Train MSE ', 'Test MSE', 'Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);