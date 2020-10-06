clear all
close all
clc

%%
% % Compute estimates
% Load image
% name_im = '.\Fig\mario_sznaier.jpg';  % Insert image file here
% X = im2double(rgb2gray(imread(name_im)));
% X = X-mean(X(:));

r = 10; % data matrix rank

%size of the input matrix
n = 400; % dimensionality of observation
m = 650;

vec1=randn(n,r);
vec2=randn(m,r);

X=vec1*vec2'; %data matrix
X = X / sqrt(sum(X(:).^2)); %normalize data matrix

% Delete 70% of the image pixels and add noise
SNR = 4;
noise=randn(n,m);
stdev = 1/(SNR*sqrt(sum((noise(:).^2))));
%stdev = sqrt(sqrt(sum((X(:).^2)))/(m*n*10^(SNR/10)));
Z=X+stdev*noise;

N_del =round(0.7*m*n); 
idx_1 = randperm(n*m);
idx_2 = idx_1(1:N_del);
Z(idx_2)=0;

%Compute delta
[~,tmp,~] = svd_thin(X);
tmp=diag(tmp);
delta = sum(tmp);

% Initialize parameters
O=Z;  % Observation
S=idx_1(N_del+1:end); %index of kown values

max_num_iter = 200;
gamma_1 = [inf, 1];
gamma_2 = [inf, 1];

for flag_in_face = 0:1
    
    %choose IF_FW_SVD_update or IF_Frank_Wolfe
    [ Z, rank_h, error_h ] = IF_FW_SVD_update( O, S, delta,  gamma_1(flag_in_face+1), gamma_2(flag_in_face+1),  max_num_iter );
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

%print('..\report\Fig\Low_Rank_Synth','-deps');

%%
% % Sweep parameters
% name_im = {'.\Fig\mario_sznaier.jpg','.\Fig\cat_img.jpg','.\Fig\dog_img.jpg'};
% 
% flag_in_face = 1;
% max_num_iter = 200;
% 
% Error_h = zeros(10,10);
% Rank_h = zeros(10,10);
% for idx = 1:length(name_im)
%     Error = zeros(10,10);
%     Rank = zeros(10,10);
%     if idx == 2
%         X = im2double((imread(name_im{idx})));
%     else
%         X = im2double(rgb2gray(imread(name_im{idx})));
%     end
%    
%     [m,n]=size(X);  % size of the input matrix
%     
%     % Delete %80 of the image pixels and add noise
%     n_del = int32((size(X,1)*size(X,2))*0.8);
%     
%     SNR = 15;
%     stdev = sqrt(sum(abs(X(:)))/(m*n*10^(SNR/10)));
%     Z = X + stdev * randn(size(X));
%     
%     
%     idx_1 = randi(m,n_del,1);
%     idx_2 = randi(n,n_del,1);
%     
%     for i = 1:n_del
%         Z(idx_1(i),idx_2(i)) = 0;
%     end
%     
%     [~,tmp,~] = svd_thin(Z);
%     tmp=diag(tmp);
%     
%     delta = sum(tmp)*0.8;
%     
%     % Initialize parameters
%     O=Z;  % Observation
%     S=find(Z~=0); %index of kown values
%     % Helper variables
%     i = 1;
%     j = 1;
%     cnt = 1;
%     
%     for gamma_1 = 0:0.1:0.9
%         i = 1;
%         for gamma_2 = gamma_1:(1-gamma_1)/9:1
%             [ Z, rank_h, error_h ] = nuclear_fw( O, S, delta,  gamma_1, gamma_2,  max_num_iter , flag_in_face , X );
%             Error(i,j) = error_h(end);
%             Rank(i,j) = rank_h(end);
%             i = i+1;
%             cnt = cnt+1;
%         end
%         j = j+1;
%     end
%     
%     Error_h=Error_h+Error;
%     Rank_h=Rank_h+Rank;
%     
% end
% 
% Error_h =Error_h/idx;
% Rank_h =Rank_h/idx;
% figure();
% subplot(2,1,1)
% imagesc(Error_h)
% title('Average MSE for Parameters','FontSize',FS2)
% xlabel('\gamma_2[n]', 'FontSize', FS1)
% ylabel('\gamma_1[n]', 'FontSize', FS1)
% pbaspect([1 1 1])
% colorbar
% box off
% 
% subplot(2,1,2)
% imagesc(Rank_h)
% title('Average Rank for Parameters','FontSize',FS2)
% xlabel('\gamma_2[n]', 'FontSize', FS1)
% ylabel('\gamma_1[n]', 'FontSize', FS1)
% pbaspect([1 1 1])
% colorbar
% box off
% %print('..\report\Fig\Full_Rank_Sp','-deps');

