function [y,dzdw]=nn_filtResNormt(x,W,dzdy,varargin)

%nn_filtResNormt :: Adjoint of CNN Filter Response Normalization (FRN)
%   Y = filtResNormt(X,W) computes the adjoint Filter Response
%   Normalization (FRNT) operator. If X is of size H x W x M x N and
%   W is of size M x K then Y is of size H x W x K x N and the FRNT
%   operation is defined as :
%
%                  M
%     Y(i,j,k,n) = S W(m,k)*X(i,j,m,n)
%                  m=1
%                   -----------------
%                      ||W(m,:)||_2
%
%   [DZDX,DZDW] = nn_filtResNormt(X, W, DZDY) computes the derivatives of
%   the block projected onto DZDY. DZDX, DZDW, and DZDY have the same
%   dimensions as X, W, and Y respectively.
%
%                     K
%     DZDX(i,j,m,n) = S W(m,k)*DZDY(i,j,k,n)
%                    k=1
%                       -----------------
%                          ||w(m,:)||_2
%
%
%                i=H,j=W,n=N
%     DZDW(m,k) =   S     X(i,j,m,n)*DZDY(i,j,k,n)
%                i=1,j=1,n=1
%
%                      1               W(m,:)^T*W(m,:)
%     DZDW(m,:)^T = ----------- *(I - ---------------- ) DZDW(m,:)^T
%                   ||W(m,:)||_2       ||W(m,:)||_2^2
%
%
%  nn_filtResNormt(...,'OPT',VALUE,...) takes the following options:
%
%   'derParams' :: if is set to false then in the backward step dzdw is not
%   computed. (Default value : true)
%
%   `cuDNN`:: {'CuDNN'} | 'NoCuDNN'
%     It indicates whether CuDNN will be used or not during the computation
%     of the convolutions.
%
% stamatis@math.ucla.edu, 03/03/2016.

opts.cuDNN='cuDNN';
opts.derParams=true;
opts=vl_argparse(opts,varargin);

% sz=size(x);
% if numel(sz < 4)
%   sz=[sz ones(1,4-numel(sz))];
% end

szW=size(W);
pdiv=@(x,y)x./y;
ptimes=@(x,y)x.*y;

Wnorm=sqrt(sum(W.^2,2));
W=bsxfun(pdiv,W,Wnorm);% W(m,:)=W(m,:)/norm(w(W,:))

if nargin <= 2 || isempty(dzdy)
  dzdw=[];
  y=vl_nnconv(x,reshape(W,[1 1 szW(1) szW(2)]),[],opts.cuDNN);
else
  y=vl_nnconv(dzdy,reshape(W',[1 1 szW(2) szW(1)]),[],opts.cuDNN);
  
  if opts.derParams && nargout > 1
    
    dzdw=zeros(szW,'like',W);
    for k=1:szW(2)
      % compute dzdw
      dzdw(:,k) = sum(sum(sum(bsxfun(ptimes,x,dzdy(:,:,k,:)),4),2),1);
%       tmp=bsxfun(ptimes,x,dzdy(:,:,k,:));
%       tmp=permute(tmp,[1 2 4 3]);
%       dzdw(:,k)=sum(reshape(tmp,[],sz(3)),1);
    end
    
    dzdw=bsxfun(pdiv,dzdw-bsxfun(ptimes,W,sum(W.*dzdw,2)),Wnorm);
  else
    dzdw = [];
  end
  
end