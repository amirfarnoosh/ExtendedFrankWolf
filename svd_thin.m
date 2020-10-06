function [u,d,v]=svd_thin(x)
% Calculates thin svd of a matrix, ignores close to zero singular values
% Args:
%   x: Input matrix
% Return:
%   u: left hand singular vectors
%   d: singular values
%   v: right hand singular vectors

[u,d,v]=svd(x);
idx=find(diag(d)>1e-6);
u=u(:,idx);
v=v(:,idx);
d=d(idx,idx);

end
