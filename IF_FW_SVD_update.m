function [ Z, Z_rank, err ] = IF_FW_SVD_update( Z, S, delta, gamma1, gamma2,  max_number_iter )

% Frank wolfe algorithm for nuclear norm minimization with SVD updates
% Args:
%   Z(matrix[mxn]) : Observation matrix
%   S(column vector) index of known values
%   delta(float) : nuclear norm upper bound
%   gamma_1,gamma_2(float) : hyperparameters for in face steps
    %set gamma_1 & gamma_2 to Inf for regular frank wolfe
%   max_number_iter(int) : maximum number of iterations

% Return:
%   Z(matrix[mxn]) : Minimum rank estimate
%   Z_rank(vec[max_number_iter x 1]) : rank of z at each iteration
%   err(vec[max_number_iter x 1]) : MSE at each iteration

%%
xij = Z(S);  %known values

%xij=xij*sqrt(1/sum(xij.^2));

[m,n]=size(Z);

%S %index for know values
%xij= %known values

niter=max_number_iter;

%Objective function 1/2 * || zij-xij ||_2^2
f=@(zij) 1/2*sum((zij-xij).^2); 

fGrad=zeros(m,n); %initialize gradient
Z=zeros(m,n);

L=1; %Lipschitz constant
dm=2*delta; %diameter of nuclear norm ball
L_bar=L;
dm_bar=dm;

fGrad(S)=(Z(S)-xij);
[u,~,v]=svds(fGrad,1);

Z=-delta*u*v'; % initialize solution
U=u;V=-v;D=delta; % initialize singular value decomposition
B=max(f(0)+trace(fGrad'*Z),0); % initialize lower bound

%%
cnt=0;
wb = waitbar(0,'Wait!');
for k=1:niter
    waitbar(k/niter);
    
    %compute gradient
    fGrad(S)=(Z(S)-xij);
    
    %compute largest left & right singular vectors
    [u,~,v]=svds(fGrad,1); 
    
    %In-face
    if sum(diag(D))<0.98*delta % If Z \in int(B) or face(B)
        %Z \in int(B)
        Z_hat=delta*u*v'; %subproblem 1
        %binary search for alpha_stop
        alpha_stop=binary_search(U,D,V,u,v,delta);
        
    else
        %Z \in face(B)
        G=1/2*(V'*fGrad'*U+U'*fGrad*V);
        [ug,~]=eigs(G,1);
        ug=ug/norm(ug);
        Z_hat=U*delta*(ug*ug')*V'; %subproblem 1
        alpha_stop=min(abs(delta*ug'*D^-1*ug-1)^-1,1);
    end
    d=Z-Z_hat; %In-Face direction
        
    % Calculate full step and minor step endpoints
    Z_B=Z+alpha_stop*d; %full step

    beta=min(-trace(fGrad'*d)/(L_bar*norm(d,'fro')^2),alpha_stop);
    Z_A=Z+beta*d; %beta [0,alpha_stop] %minor step

    %Choose between Z_B or Z_A or regular frank-wolfe 
    if 1/(f(Z_B(S))-B)>=1/(f(Z(S))-B)+gamma1/(2*L_bar*dm_bar^2)
        %go to lower dimensional face
        Z=Z_B;
        if sum(diag(D))<0.98*delta  % If Z \in int(B) or face(B)
            [U,D,V]=svd_update(U,(1+alpha_stop)*D,V,-alpha_stop*delta*u,v);
        else
            [R,D,R]=svd_thin((1+alpha_stop)*D-alpha_stop*delta*(ug*ug'));
            U=U*R;
            V=V*R;
        end

    elseif 1/(f(Z_A(S))-B)>=1/(f(Z(S))-B)+gamma2/(2*L_bar*dm_bar^2)
        %stay in face
        Z=Z_A;
        if sum(diag(D))<0.98*delta % If Z \in int(B) or face(B)
            [U,D,V]=svd_update(U,(1+beta)*D,V,-beta*delta*u,v);
        else
            [R,D,R]=svd_thin((1+beta)*D-beta*delta*(ug*ug'));
            U=U*R;
            V=V*R;
        end

    else
        %regular frank-wolfe 
        cnt=cnt+1;
        Z_tilda=-delta*u*v'; %regular frank-wolfe direction
        Bw=f(Z(S))+trace(fGrad'*(Z_tilda-Z)); %update lower bound
        B=max(B,Bw); 
        alpha=min(trace(fGrad'*(Z-Z_tilda))/(L_bar*norm(Z-Z_tilda,'fro')^2),1);
        Z=Z+alpha*(Z_tilda-Z);
        [U,D,V]=svd_update(U,(1-alpha)*D,V,-alpha*delta*u,v);

    end
 
    
    Z_rank(k)=length(D);
    err(k)=f(Z(S));
    
end
close(wb)
