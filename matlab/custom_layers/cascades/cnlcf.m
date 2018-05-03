function [y,dzdwh,dzdwht,dzdg,dzdgt,dzdw,dzdl,J,M] = cnlcf(x,Obs,h,wh,wht,...
  g,gt,rbf_weights,rbf_means,rbf_precision,lambda,dzdy,varargin)
%(Color Non-Local Collaborative Filtering) 
%
% Each nlcf stage consists of the following layers:
% 1) NN_CPGCF 2) NN_CLIP 3) NN_SHRINK_LUT 4) NN_CPGCFT 5) NN_DATAFIDELITY
%
%  If any of wht and gt are empty then the 1st and 4th layers share the  
%  same parameter wh or/and g.
%
%
%  x      : input of the current stage
%  Obs    : input of the first stage of the network.
%  lambda : parameter for NN_DATAFIDELITY
%
% If opts.padSize~=[0,0,0,0] then the NLCF stage consists of
% the following layers
%
% 1) NN_PAD 2) NN_CPGCF 3) NN_CLIP 4) NN_SHRINK_LUT 5) NN_CPGCFT 
% 6) NN_PADT 7) NN_DATAFIDELITY
%
%  If any of wht and gt are empty then the 2nd and 5th layers share the  
%  same parameter wh or/and g.
%
%   Y = CNLCF(X,Obs,H,WH,WHT,G,GT,RBF_WEIGHTS,MEANS,PRECISION,LAMBDA)
%
%   In the forward mode X is of size H x W x 1 x N (K: number of channels,
%   N: number of images). H, WH, G are the inputs to the
%   layers nn_pgcf, nn_pgcft. RBF_weights
%   is a matrix of size B x M where M is the number of mixture components
%   in the RBF-mixture, RBF_means is of size M x 1 and rbf_precision is a
%   scalar. Lambda is a scalar that is a learnable parameter of the 
%   NN_DATAFIDELITY layer.
%
%   If ConserveMemory is set to false (see below) then Y is a 4 x 1 cell
%   array which keeps the outputs of some of the internal layers, necessary
%   for the backward mode.
%
%   [DZDX, DZDWH, DZDWHT, DZDG, DZDGT, DZDW, DZDL] = CNLCF(X, Obs, H, ...
%   WH, WHT, G, GT, RBF_WEIGHTS, MEANS, PRECISION, LAMBDA, DZDY) 
%   computes the derivatives of the stage projected onto DZDY. DZDX, 
%   DZDWH, DZDWHT, DZDG, DZDGT, DZDW, DZDL and DZDY have the
%   same dimensions as X, WH, WHT,G, GT, RBF_WEIGHTS, LAMBDA and Y{4} 
%   respectively. In the X is a cell array (3,1) where X{1} = X_in,
%   X{2} = {Y{3},Y{2}}, X{3} = Y{1}, where Y is the output of the forward 
%   mode with the conserveMemory option set to false and X_in is the 
%   input of the forward mode.
%
%   CNLCF(...,'OPT',VALUE,...) takes the following options:
%
%   `Jacobian`:: J (see RBFSHRINK)
%
%   'Idx' :: Idx is an array of the same size as the output of the
%       NN_FiltResNorm layer and can be computed using the function
%       Idx=misc.gen_idx_4shrink(size(x));
%
%   `Nbrs_idx` :: It is the output of misc.patchMatch applied on
%       vl_nnpad(x,opts.padSize);
%
%   `Stride`:: 1
%       The output stride or downsampling factor. If the value is a
%       scalar, then the same stride is applied to both vertical and
%       horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%       allows specifying different downsampling factors for each
%       direction.
%
%   `Nbrs`:: 4
%       The number of closest neighbors used in the block-matching layer of
%       the non-local range convolution.
%
%   `searchwin`:: [15 15]
%       The half-dimensions of the search window in which the closest
%       neighbors are searched for.
%
%   `patchDist`:: {'euclidean'} | 'abs'
%       It specifies the type of the distance that will be used for the image
%       patch similarity check.
%
%   `Padsize:: Specifies the amount of padding of the input X as
%       [TOP, BOTTOM, LEFT, RIGHT].
%
%   `transform` ::  If set to true then the patch similarity takes place
%       in the gradient domain instead of the image domain. (Default : false)
%
%   `useSep` :: If set to true then the separable knnpatch is employed to
%       find the closest neighbors.
%
%   `sorted` :: If set to true then the neighbors are sorted according to
%       their distance from the patch of interest.
%
%   ATA:: @(x)x (Used in CNN_SUM)
%       Allows to define a function handle which corresponds to the operation
%       A^T(A(x)) where A is a linear operator and A^T is its adjoint.
%
%   AT:: @(x)x (Used in CNN_SUM)
%       Allows to define a function handle which corresponds to the operation
%       A^T(x) where A^T is the adjoint of a linear operator A.
%
%   'learningRate' :: 6x1 vector. if an element is set to zero then 
%       in the backward step the derivatives of the corresponding weights 
%       (dzdwh, dzdwht, dzdg, dzdgt, dzdw, dzdl) are not computed. 
%       (Default value : [1,1,1,1,1,1])
%
%   `ConserveMemory`:: false | true
%       If set to true saves memory by not storing the intermediate results
%       from the different layers. If used in training then the value
%       should be set to false.

% s.lefkimmiatis@skoltech.ru, 30/06/2017

% Example
% x = single(imread('/Users/stamatis/Documents/MATLAB/Data/BSDS500/color/102061.jpg'));
% support = [5 5]; numFilters=24; padSize = [2,2,2,2]; stride=[1,1];
% Nbrs = 4; searchwin = [15 15];
% stdn = 25; y = x+stdn*randn(size(x),'like',x);
% [idx, dist] = misc.patchMatch(nn_pad(x,padSize),'stride',stride, ...
%   'Nbrs',Nbrs,'searchwin',searchwin,'patchsize',support);
% lambda = 0;
% cid = misc.getClass(x);
% h = misc.gen_dct2_kernel(support,'classType',cid,'gpu',false);
% h = h(:,:,:,2:end);
% g = randn(1,Nbrs,'like',x);
% wh = randn(numFilters,cid);
% rbf_means=cast(-100:4:100,cid); rbf_precision = 4;
% rbf_weights = randn(numFilters,numel(rbf_means),cid);
% data_mu=cast(-104:0.1:104,cid);
% data_mu=bsxfun(@minus,data_mu,cast(rbf_means(:),cid));
% [xe,~,~,~,~,~,~,J,M] = cnlcf(y,y,h,wh,[],g,[],rbf_weights,rbf_means,...
%  rbf_precision,lambda,[]'stride',stride,'padSize',padSize,'lb',-100,...
%  'ub',100,'Nbrs_idx',idx,'conserveMemory',false,'data_mu',data_mu);
%
% dzdy = randn(size(xe{end}),'like',x);
% [dzdx,dzdwh,~,dzdg,~,dzdw,dzdl] = cnlcf({y,{xe{3},xe{2}},xe{1}},...
%   y,h,wh,[],g,[],rbf_weights,rbf_means,rbf_precision,lambda,dzdy,...
%   'stride',stride,'padSize',padSize,'lb',-100,'ub',100,...
%   'Nbrs_idx',idx,'Jacobian',J,'clipMask',M,'data_mu',data_mu);

opts.stride = [1,1];
opts.padSize = [0,0,0,0];
opts.cuDNN = 'cuDNN';
opts.Jacobian = [];
opts.clipMask = [];
opts.Idx=[];
opts.conserveMemory = false;
opts.first_stage = false; % Denotes if this the first stage of the network.
opts.learningRate = [1,1,1,1,1,1];
opts.data_mu = [];
opts.step = [];
opts.origin = [];
opts.shrink_type = 'identity';
opts.ATA = @(x)x;
opts.AT = @(x)x;
opts.identity = false;
opts.lb = -inf;
opts.ub = inf;

% Params for patchMatch
opts.searchwin = [15,15];
opts.patchDist = 'euclidean';
opts.transform = false;
opts.useSep = true;
opts.sorted = true;
opts.Wx = []; % kernel for weighting the patch elements.
opts.Wy = [];
opts.Nbrs_idx = [];
opts.Nbrs_weights = [];
opts.PatchWeighting = false;
opts.Nbrs = [];
%-----------------------------
opts = vl_argparse(opts,varargin);

if numel(opts.learningRate) ~= 6
  opts.learningRate = repmat(opts.learningRate(1),6,1);
end

switch opts.shrink_type
  case 'identity'
    Shrink = @nn_grbfShrink_lut;
  case 'residual'
    Shrink = @nn_grbfResShrink_lut;
  otherwise
    error('cnlcf :: Unknown type of RBF shrinkage.');
end

usePad = (sum(opts.padSize) ~= 0);

if isempty(opts.Nbrs) 
  if isempty(opts.Nbrs_idx)
    error('cnlcf :: The number of closest neighbors must be specified.');
  else
    opts.Nbrs = size(opts.Nbrs_idx,3);
  end
end

numFilters = size(wh,1);

assert(numFilters == size(rbf_weights,1), ['Invalid input for ' ...
  'wh - dimensions mismatch.']);
assert( isempty(wht) || all(size(wh) == size(wht)), ...
  'Invalid input for wht - dimensions mismatch.');

weightSharing = isempty(wht);% If wht is empty then pgcf and pgcft
% share the same spatial weights.

assert( numel(g) == opts.Nbrs || numel(g) == numFilters*opts.Nbrs, ...
  'Invalid input for g - dimensions mismatch.')

assert( isempty(gt) || all(size(gt) == size(g)), ...
  'Invalid input for gt - dimensions mismatch.')

GroupWeightSharing = isempty(gt); 



if nargin < 12 || isempty(dzdy)
  dzdwh = []; dzdwht = [];
  dzdg = []; dzdgt = [];
  dzdw = []; dzdl = []; J = []; M = [];
  
  Nc = size(x,3);
  
  if usePad
    x = nn_pad(x,opts.padSize);
  end
  
  % Block-matching
  if isempty(opts.Nbrs_idx)
    [opts.Nbrs_idx,opts.Nbrs_weights] = misc.patchMatch(x(:,:,1,:),'patchSize', ...
      [size(h,1) size(h,2)],'searchwin',opts.searchwin,'stride',...
      opts.stride,'Nbrs',opts.Nbrs,'patchDist',opts.patchDist,...
      'transform',opts.transform,'cuDNN',opts.cuDNN,'useSep',...
      opts.useSep,'sorted', opts.sorted,'Wx',opts.Wx,'Wy',opts.Wy);
    
    if opts.PatchWeighting
      opts.Nbrs_weights = bsxfun(@rdivide,opts.Nbrs_weights,max(opts.Nbrs_weights,[],3));
      opts.Nbrs_weights(isnan(opts.Nbrs_weights)) = 0;
      opts.Nbrs_weights = exp(-opts.Nbrs_weights);
      opts.Nbrs_weights = bsxfun(@rdivide,opts.knn_weights,sum(opts.Nbrs_weights,3));
    else
      opts.Nbrs_weights = [];
    end  
  end  
  
  if opts.PatchWeighting && isempty(opts.Nbrs_weights)
    warning(['Weighting of similar patches has been chosen ' ...
      'but the weights have not been provided']);
  end
  
  if opts.conserveMemory
    y = nn_cpgcf(x,h,wh,opts.Nbrs_weights,g,opts.Nbrs_idx,[],'stride',...
      opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',true);
    
    if nargout > 8
      [y,M] = nn_clip(y,opts.lb,opts.ub);
    else
      y = nn_clip(y,opts.lb,opts.ub);
    end
    
    if nargout > 7
      J = zeros(size(y),'like',y);
      for ch = 1:Nc
        idx_ch = (1:numFilters)+(ch-1)*numFilters;
        [y(:,:,idx_ch,:),~,J(:,:,idx_ch,:)] = Shrink(y(:,:,idx_ch,:), ...
          rbf_weights(:,:,ch), rbf_means,rbf_precision,[],'Idx',...
          opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
          opts.origin);
      end
    else
      for ch = 1:Nc
        idx_ch = (1:numFilters)+(ch-1)*numFilters;
        y(:,:,idx_ch,:) = Shrink(y(:,:,idx_ch,:),rbf_weights(:,:,ch), ...
          rbf_means,rbf_precision,[],'Idx',opts.Idx,'data_mu', ...
          opts.data_mu,'step',opts.step,'origin',opts.origin);
      end
    end
    
    if weightSharing && GroupWeightSharing
      y = nn_cpgcft(y,h,wh,opts.Nbrs_weights,g,opts.Nbrs_idx,[],'stride',...
        opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',true);
    elseif weightSharing && ~GroupWeightSharing
      y = nn_cpgcft(y,h,wh,opts.Nbrs_weights,gt,opts.Nbrs_idx,[],'stride',...
        opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',true);      
    elseif ~weightSharing && GroupWeightSharing
      y = nn_cpgcft(y,h,wht,opts.Nbrs_weights,g,opts.Nbrs_idx,[],'stride',...
        opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',true);            
    else
      y = nn_cpgcft(y,h,wht,opts.Nbrs_weights,gt,opts.Nbrs_idx,[],'stride',...
        opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',true);            
    end
    
    if usePad
      y = nn_padt(y,opts.padSize);
      y = nn_L2ProxGradUpdate(y,...
        x(opts.padSize(1)+1:end-opts.padSize(2),opts.padSize(3)+1:...
        end-opts.padSize(4),:,:),Obs,lambda,[],'ATA',opts.ATA,'AT',...
        opts.AT,'identity',opts.identity);
    else
      y = nn_L2ProxGradUpdate(y,x,Obs,lambda,[],'ATA',opts.ATA,'AT',...
        opts.AT,'identity',opts.identity);
    end 
    
  else
    y=cell(4,1);
    
    y{1} = nn_cpgcf(x,h,wh,opts.Nbrs_weights,g,opts.Nbrs_idx,[],'stride',...
      opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',false);
    
   if nargout > 8
      [y{1}{3},M] = nn_clip(y{1}{3},opts.lb,opts.ub);
    else
      y{1}{3} = nn_clip(y{1}{3},opts.lb,opts.ub);
    end 
        
    if nargout > 7
      J = zeros(size(y{1}{3}),'like',y{1}{3});
      for ch = 1:Nc
        idx_ch = (1:numFilters)+(ch-1)*numFilters;
        [y{2}(:,:,idx_ch,:),~,J(:,:,idx_ch,:)] = Shrink(...
          y{1}{3}(:,:,idx_ch,:),rbf_weights(:,:,ch),rbf_means,...
          rbf_precision,[],'Idx',opts.Idx,'data_mu',opts.data_mu,'step',...
          opts.step,'origin',opts.origin);
      end
    else
      for ch = 1:Nc
        idx_ch = (1:numFilters)+(ch-1)*numFilters;
        y{2}(:,:,idx_ch,:) = Shrink(y{1}{3}(:,:,idx_ch,:), ...
          rbf_weights(:,:,ch),rbf_means,rbf_precision,[],'Idx',opts.Idx, ...
          'data_mu',opts.data_mu,'step',opts.step,'origin',opts.origin);
      end
    end
    
    if weightSharing && GroupWeightSharing
      y{3} = nn_cpgcft(y{2},h,wh,opts.Nbrs_weights,g,opts.Nbrs_idx,[],'stride',...
        opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',false);
    elseif weightSharing && ~GroupWeightSharing
      y{3} = nn_cpgcft(y{2},h,wh,opts.Nbrs_weights,gt,opts.Nbrs_idx,[],'stride',...
        opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',false);      
    elseif ~weightSharing && GroupWeightSharing
      y{3} = nn_cpgcft(y{2},h,wht,opts.Nbrs_weights,g,opts.Nbrs_idx,[],'stride',...
        opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',false);            
    else
      y{3} = nn_cpgcft(y{2},h,wht,opts.Nbrs_weights,gt,opts.Nbrs_idx,[],'stride',...
        opts.stride,'Nbrs',opts.Nbrs,'cuDNN',opts.cuDNN,'conserveMemory',false);            
    end
    
    if usePad
      y{3}{2} = nn_padt(y{3}{2},opts.padSize);
      y{4} = nn_L2ProxGradUpdate(y{3}{2}, ...
        x(opts.padSize(1)+1:end-opts.padSize(2),opts.padSize(3)+1:...
        end-opts.padSize(4),:,:),Obs,lambda,[],'ATA',opts.ATA,'AT',...
        opts.AT,'identity',opts.identity);
    else
      y{4} = nn_L2ProxGradUpdate(y{3}{2}, ...
        x(opts.padSize(1)+1:end-opts.padSize(2),opts.padSize(3)+1:...
        end-opts.padSize(4),:,:),Obs,lambda,[],'ATA',opts.ATA,'AT',...
        opts.AT,'identity',opts.identity);
    end   
    
    y{3} = y{3}{1}; % We don't need to store y{3}{2}
    
  end
  
else
  J = []; M = [];
  Nc = size(dzdy,3);
  dzdw = zeros(size(rbf_weights),'like',rbf_weights);
  
  if opts.first_stage
    % In this case we don't care about correctly
    % computing dzdx where x the input of the stage. We also do not have to
    % propagate back until the input of the stage. Insted we propagate only
    % until the input of the NN_PGCF
    [y,dzdl] = nn_L2ProxGradUpdate([], x{1}, Obs, lambda, dzdy,'ATA', ...
      opts.ATA,'AT', opts.AT, 'identity', opts.identity, 'derParams', ...
      logical(opts.learningRate(6)));    
  else
    [y,dzdl,dzdx] = nn_L2ProxGradUpdate([], x{1}, Obs, lambda, dzdy, ...
      'ATA', opts.ATA, 'AT', opts.AT, 'identity', opts.identity, ...
      'derParams', logical(opts.learningRate(6)));
  end
  clear dzdy;
  x{1} = [];
  
  if usePad
    y = nn_padt([],opts.padSize,y);
  end

  if weightSharing && GroupWeightSharing
    [y,dzdwht,dzdgt] = nn_cpgcft(x{2},h,wh,opts.Nbrs_weights,g, ...
      opts.Nbrs_idx,y,'stride',opts.stride,'cuDNN',opts.cuDNN, ...
      'Nbrs',opts.Nbrs,'derParams',logical(opts.learningRate([1,3])));
  elseif weightSharing && ~GroupWeightSharing
    [y,dzdwht,dzdgt] = nn_cpgcft(x{2},h,wh,opts.Nbrs_weights,gt, ...
      opts.Nbrs_idx,y,'stride',opts.stride,'cuDNN',opts.cuDNN, ...
      'Nbrs',opts.Nbrs,'derParams',logical(opts.learningRate([1,4])));    
  elseif ~weightSharing && GroupWeightSharing
    [y,dzdwht,dzdgt] = nn_cpgcft(x{2},h,wht,opts.Nbrs_weights,g, ...
      opts.Nbrs_idx,y,'stride',opts.stride,'cuDNN',opts.cuDNN, ...
      'Nbrs',opts.Nbrs,'derParams',logical(opts.learningRate([2,3])));
  else
    [y,dzdwht,dzdgt] = nn_cpgcft(x{2},h,wht,opts.Nbrs_weights,gt, ...
      opts.Nbrs_idx,y,'stride',opts.stride,'cuDNN',opts.cuDNN, ...
      'Nbrs',opts.Nbrs,'derParams',logical(opts.learningRate([2,4])));    
  end
  x{2} = [];
    
  
  for ch = Nc:-1:1 % This way no error will occur at the dzdw(:,:,ch) 
    % assignment if opts.learningRate(5) is zero.
    idx_ch = (1:numFilters)+(ch-1)*numFilters;
    if isempty(opts.Jacobian)
      [y(:,:,idx_ch,:), dzdw(:,:,ch)] = Shrink(x{3}{3}(:,:,idx_ch,:), ...
        rbf_weights(:,:,ch),rbf_means,rbf_precision,y(:,:,idx_ch,:),...
       'Jacobian',[],'Idx',opts.Idx,'derParams',...
        logical(opts.learningRate(5)),'data_mu',opts.data_mu,'step',...
        opts.step,'origin',opts.origin);
    else
      [y(:,:,idx_ch,:), dzdw(:,:,ch)] = Shrink(x{3}{3}(:,:,idx_ch,:), ...
        rbf_weights(:,:,ch),rbf_means,rbf_precision,y(:,:,idx_ch,:),...
       'Jacobian',opts.Jacobian(:,:,idx_ch,:),'Idx',opts.Idx,'derParams',...
        logical(opts.learningRate(5)),'data_mu',opts.data_mu,'step',...
        opts.step,'origin',opts.origin);      
    end
  end
  x{3}(3) = [];
  
  y = nn_clip([],opts.lb,opts.ub,y,'mask',opts.clipMask); 
  
  [y,dzdwh,dzdg] = nn_cpgcf(x{3}(2:-1:1),h,wh,opts.Nbrs_weights,g, ...
    opts.Nbrs_idx,y,'stride',opts.stride,'cuDNN',opts.cuDNN, ...
    'Nbrs',opts.Nbrs,'derParams',logical(opts.learningRate([1,3])));
  
  clear x;
  
  if weightSharing && GroupWeightSharing
    dzdwh = dzdwh + dzdwht; 
    dzdwht = [];
    dzdg = dzdg + dzdgt; 
    dzdgt = [];
  elseif weightSharing && ~GroupWeightSharing
    dzdwh = dzdwh + dzdwht; 
    dzdwht = [];
  elseif ~weightSharing && GroupWeightSharing
    dzdg = dzdg + dzdgt; 
    dzdgt = [];
  end
  
  % If this is the first stage of the network then we don't need to
  % correctly compute dzdx and therefore we save computations.
  if ~opts.first_stage
    if usePad
      y = nn_pad([],opts.padSize,y);
    end
    y = y+dzdx;
  else
    y = [];
  end
  
end

