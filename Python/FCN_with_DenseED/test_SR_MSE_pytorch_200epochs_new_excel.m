clc 
close all
clear variables

%date:6th Feb 2021
%VM
font = 14;
linewidth = 2;
format long;
%addpath('/Users/varunmannam/Desktop/Spring21/Research_S21/Feb21/SPIE_SR/old_SR_files/mse_plots/');
addpath('/Users/varunmannam/Desktop/Spring21/Research_S21/Feb21/0702/SR_new_results/mse_loss/'); %new simulations
lr1 = load('DenseED_test_RMSE_0702_2021_20210207_000054_vm41.txt'); %new result with VM4 SR result MSE 
lr2 = load('DenseED_test_RMSE_0702_2021_20210207_000620_vm521.txt'); %new result with VM52 SR result MSE 
lr3 = load('DenseED_test_RMSE_0702_2021_20210207_014432_vm6141.txt'); %new result with VM614 SR result MSE  


epochs = 200;
x=[1:epochs]';

figure, 
plot(x,lr1,'b--', 'Linewidth', linewidth);
hold on
plot(x,lr2,'r--', 'Linewidth', linewidth);
plot(x,lr3,'g--', 'Linewidth', linewidth);
xlabel('Epochs');
ylabel('MSE loss');
%title('training loss Unet nbn Keras');
legend('Configuration: (9,18,9) with 400 images', 'Configuration: (8,8,8) with 400 images', 'Configuration: (9,18,9) with 750 images', 'Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);


figure, 
plot(x(epochs-10:epochs),lr1(epochs-10:epochs),'b-', 'Linewidth', linewidth);
hold on
plot(x(epochs-10:epochs),lr2(epochs-10:epochs),'r-', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr3(epochs-10:epochs),'g-', 'Linewidth', linewidth);

xlabel('Epochs');
ylabel('MSE loss');
%title('training loss Unet nbn Keras');
legend('Configuration: (9,18,9) with 400 images', 'Configuration: (8,8,8) with 400 images', 'Configuration: (9,18,9) with 750 images', 'Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);