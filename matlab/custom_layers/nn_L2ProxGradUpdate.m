function [y,dzdl,dzdx]=nn_L2ProxGradUpdate(r,x,Obs,lambda,dzdy,varargin)

% L2ProxGradUpdate : It performs one proximal gradient update using an
% L2-squared data fidelity term of the form D = ||AT(x)-Obs||^2. In this
% case the gradient of the data fidelity w.r.t x is Dg = ATA(x)-AT(Obs), 
% r is the gradient of the regularizer w.r.t x and lambda is the step-size
% of the proximal update.

%   Y = nn_L2ProxGradUpdate(R,X,OBS,LAMBDA) computes the output of the layer.
%   R and X are of size H x W x M x N, LAMBDA is a scalar and Obs are the
%   observations of the forward model related to the inverse problem that 
%   we want to solve.
%  
%   Y = X - lambda*(ATA(X)-AT(OBS)) - R;
%
%   [DZDR,DZDL,DZDX] = L2ProxGradUpdate(R,X,OBS,W,DZDY) computes the 
%   derivatives of the block projected onto DZDY. DZDR, DZDX, DZDL, and 
%   DZDY have the same dimensions as R, X, LAMBDA, and Y respectively.
%
%   DZDR = -DZDY
%   DZDX = DZDY - LAMBDA*ATA(DZDY)
%   DZDL =  < AT(OBS), DZDY > - < ATA(X), DZDY >
%
%
%   L2ProxGradUpdate(...,'OPT', VALUE, ...) supports the additional options:
%
%   ATA:: @(x)x
%     Allows to define a function handle which corresponds to the operation
%     A^T(A(x)) where A is a linear operator and A^T is its adjoint.
%   AT:: @(x)x
%     Allows to define a function handle which corresponds to the operation
%     A^T(x) where A^T is the adjoint of a linear operator A.
%
%   'derParams' :: if is set to false then in the backward step dzdl is not
%    computed. (Default value : true)
%

% s.lefkimmiatis@skoltech.ru, 19/04/2017.

opts.derParams = true;
opts.ATA=@(x)x;
opts.AT=@(x)x; % opts.AT corresponds to the adjoint of the forward operator 
% A which is responsible for the degradation of the measurements (e.g A can
% be a convolution operator (deconvolution problem), a mask operator
% (inpainting problem) or the identity operator (denoising) ).

% Hint :: During training we can always precompute AT(Obs) and therefore we
% can skip its computation in every training iteration. In this case
% we need to set opts.AT = @(x)x so as to ensure the correct behavior.

opts.identity = false; % If its set to true opts.ATA and opts.AT are 
% considered to be equal to the identity operator and they are ignored.

opts=vl_argparse(opts,varargin);

if opts.identity
  if nargin < 5 || isempty(dzdy)
    dzdx=[];
    dzdl=[];
    y = (1-lambda)*x + lambda*Obs - r;
  else
    y = -dzdy;  
    if nargout == 3
      if opts.derParams
        dzdl = sum(reshape((Obs-x).*dzdy,[],1));
      else
        dzdl = [];
      end
      dzdx = (1-lambda)*dzdy;
    elseif nargout == 2
      if opts.derParams
        dzdl = sum(reshape((Obs-x).*dzdy,[],1));
      else
        dzdl = [];
      end
      dzdx = [];
    else
      dzdl = [];
      dzdx = [];
    end
  end 
else
  if nargin < 5 || isempty(dzdy)
    dzdx=[];
    dzdl=[];
    y = x - lambda*(opts.ATA(x)-opts.AT(Obs)) - r;
  else
    y = -dzdy;    
    if nargout == 3
      if opts.derParams
        dzdl = sum(reshape((opts.AT(Obs)-opts.ATA(x)).*dzdy,[],1));
      else
        dzdl = [];
      end
      dzdx = dzdy-lambda*opts.ATA(dzdy);
    elseif nargout == 2
      if opts.derParams
        dzdl = sum(reshape((opts.AT(Obs)-opts.ATA(x)).*dzdy,[],1));
      else
        dzdl = [];
      end
      dzdx = [];
    else
      dzdl = [];
      dzdx = [];
    end
  end
end