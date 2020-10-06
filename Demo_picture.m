clear all
close all
clc

%%
% Compute estimates
% Load image
name_im = '.\Fig\MarioSznaier.jpg';  % Insert image file here
X = im2double(rgb2gray(imread(name_im)));
X = X-mean(X(:));

[m,n] = size(X);
N_del =round(0.5*m*n); 
idx_1 = randperm(n*m);
idx_2 = idx_1(1:N_del);
Z=X;
Z(idx_2)=0;

%Compute delta
[~,tmp,~] = svd_thin(X);
tmp=diag(tmp);
delta = sum(tmp)*0.7;

% Initialize parameters
O=Z;  % Observation
S=idx_1(N_del+1:end); %index of kown values

max_num_iter = 300;
gamma_1 = [inf, 1];
gamma_2 = [inf, 1];

for flag_in_face = 0:1
    
    %choose IF_FW_SVD_update or IF_Frank_Wolfe
    [ Z, rank_h, error_h ] = IF_FW_SVD_update( O, S, delta,  gamma_1(flag_in_face+1), gamma_2(flag_in_face+1),  max_num_iter );
    img{flag_in_face+1}=Z;
    rank_lh{flag_in_face+1} = rank_h;
    error_lh{flag_in_face+1} = error_h;
    
end

FS1=13;
FS2=15;
figure();
subplot(2,1,1)
plot((error_lh{1}),'LineWidth',2)
hold();
plot((error_lh{2}),'--','LineWidth',2)
title(strcat('Low\_Rank Image - \gamma_1= ',num2str(gamma_1(2)),' \gamma_2= ',num2str(gamma_2(2))),'FontSize',FS2)
h=legend('FW','FW-In Face');
set(h,'FontSize',FS1); 
xlabel('iterations', 'FontSize', FS1)
ylabel('objective', 'FontSize', FS1)
box off

subplot(2,1,2)
plot(rank_lh{1},'LineWidth',2)
hold();
plot(rank_lh{2},'--','LineWidth',2)
ylabel('rank', 'FontSize', FS1)
xlabel('iterations', 'FontSize', FS1)
h=legend('FW','FW-In Face');
 set(h,'FontSize',FS1); 
box off

figure()
subplot(2,2,1)
imshow(X,[]);
title('Original image');
subplot(2,2,3)
imshow(O,[]);
title('Image with missing pixels');
subplot(2,2,2)
imshow(img{1},[]);
title('Recostructed image-regular FW');
subplot(2,2,4)
imshow(img{2},[]);
title('Recostructed image-In Face FW');