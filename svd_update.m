function [u,s,v]=svd_update(U,S,V,a,b)
% Updates SVD of a matrix, where iterations are in form x_2 = x_1 + ab'
% SVD(x_2) = svd_update(SVD(x_1))
% Args:
%   U,S,V: initial singular value decomposition
%   a,b: vectors that form the rank 1 matrix
% Return:
%   u,s,v: updated singular value decomposition

r=length(S);
m=U'*a;p=a-U*m;
p_norm=norm(p);P=p/p_norm;
n=V'*b;q=b-V*n;
q_norm=norm(q);Q=q/q_norm;
K=[S,zeros(r,1);zeros(1,r),0]+[m;p_norm]*[n;q_norm]';
[uk,s,vk]=svd_thin(K);
u=[U P]*uk;
v=[V Q]*vk;

end