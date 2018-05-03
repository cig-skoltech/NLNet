function [y,dh,dg] = nn_pgcft(x,h,wh,weights,g,idx,dzdy,varargin)

% NN_PGCFT (Transpose of Patch Group Collaborative Filtering)
%
%   Y = NN_PGCF(X, H, WH, G, idx) extracts in total Np 2D-patches of size 
%   PH x PW from the input X, groups the similar patches (specified by idx) 
%   into 3D stacks (each stack has a size of PH x PW x Nbrs, where Nbrs is
%   the number of the closest neighbors to the reference patch) and 
%   performs a 2D spatial transform (R: PH*PW -> M) to each patch of the 
%   stack followed by a 1-D collaborative filtering across the group of 
%   similar patches of the stack. The size of the final output Y is 
%   H' x W' x M x N, where Np = H' x W' denotes the total number of 
%   extracted patches. 
%
%   Z = NN_PGCFT(Y, H, WH, Weights, G, idx)
%
%   X is an array of dimension H x W x 1 x N where (H,W) are
%   the height and width of the image stack and N the number of images in 
%   the stack.
%
%   H is an array of dimension PH x PW x 1 x K where (PH,PW)
%   are the filter height and width and K the number of filters in the
%   bank. 
%
%   WH is an array of size M x K (used in VL_NNCONVT_FRN)
%
%   G is an array of size either Nbrs or M*Nbrs, where Nbrs is the 
%   number of the closest neighbors used in forming a 3D stack of similar
%   patches. If numel(G)=Nbrs then during the collaborative filtering that
%   takes place in the 3rd dimension of the patch group the same weights
%   are used for each patch transform coefficient of the patch.
%
%   Idx is an array of size H'x W' x opts.Nbrs x N
%   where Np = H' x W' is the total number of patches. It is computed in 
%   the block-matching layer (misc.patchMatch).
%
%   Weights is either an empty array or an array of size 
%   H'x W' x opts.Nbrs x N where Np = H' x W' is the total number of 
%   patches. It can computed by the misc.patchMatch function.
%
%   If ConserveMemory is set to false (see below) then Y is a 3 x 1 cell 
%   array which keeps the outputs of some of the internal layers, necessary
%   for the backward mode.
%
%   [DZDX,DH,DG] = NN_PGCFT(X, H, WH, G, idx, DZDY) 
%   computes the derivatives of the block projected onto DZDY. DZDX, DH 
%   and DG have the same dimensions as X, WH and G, respectively. In 
%   the backward mode X is 2x1 cell array where X{1}=Y{1} and X{2}=X.
%
%   NN_PGCF(..., 'option', value, ...) takes the following options:
%
%   `Stride`:: 4
%     The output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical and
%     horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%     allows specifying different downsampling factors for each
%     direction.
%
%   `Nbrs`:: 8
%     The number of closest neighbors used in the block-matching layer of
%     the non-local range convolution.
%
%   'derParams' :: if is set to false then in the backward step dzdw is not
%     computed. (Default value : true)
%
%   `cuDNN`:: {'CuDNN'} | 'NoCuDNN' 
%     It indicates whether CuDNN will be used or not during the computation 
%     of the convolutions.
%
%   `ConserveMemory`:: false | true
%       If set to true saves memory by not storing the intermediate results
%       from the different layers. If used in training then the value 
%       should be set to false.
%

% s.lefkimmiatis@skoltech.ru, 09/07/2016

% patchSize=[5,5];stride=[1,1];Nbrs=5;searchwin=[12 12]; 
% useGPU = false;
% h=misc.gen_dct2_kernel(patchSize,'classType',misc.getClass(x),'gpu',useGPU);
% h = h(:,:,:,2:end);
% wh = randn(size(h,4),'like',h);
% g = randn(Nbrs,1,'like',h);
% [idx, dist] = misc.patchMatch(x,'stride',stride,'Nbrs',Nbrs,'searchwin',searchwin,...
%   'patchsize',patchSize);
% weights = bsxfun(@times,dist,1./max(dist,[],3)); 
% weights = bsxfun(@times,exp(-weights),1./sum(exp(-weights),3));
%
%  % Forward Mode
% y = nn_pgcf(x,h,wh,weights,g,idx,[],'stride',stride,'Nbrs',Nbrs, ...
% 'cuDNN','cuDNN','conserveMemory',false);
%
% % Backward Mode
% input = {y{2},y{1}};
% dzdy = randn(size(y{end}),'like',y{end});
% [dzdx,dh,dg] = nn_pgcf(input,h,wh,weights,g,idx,dzdy,'stride',stride,'Nbrs',Nbrs);

opts.stride = [1,1];
opts.Nbrs = 8; % Number of neighbors for the block-matching layer.
opts.cuDNN = 'cuDNN';
opts.derParams = [true,true];
opts.conserveMemory = false;
opts = vl_argparse(opts,varargin);

numFilters = size(wh,1);

if numel(opts.derParams) <  2
  opts.derParams = [opts.derParams,repmat(opts.derParams(1),1,...
    2-numel(opts.derParams))];
end

assert(size(wh,2) == size(h,4), ['nn_pgcft:: Invalid input for ' ...
  'wh - dimensions mismatch.']);

assert( isempty(weights) || all(size(weights) == size(idx)), ...
  'nn_pgcft:: Invalid input for weights - dimensions mismatch.');

if numel(opts.stride) == 1
  opts.stride = opts.stride*[1,1];
end

if numel(g) ~= opts.Nbrs && numel(g) ~= opts.Nbrs*numFilters
  error(['nn_pgcft:: The number of the neighbor-filter coefficients ' ...
  'must be either equal to the number of the closest neighbors or equal'...
  ' to the number of closest neighbors times the number of the feature' ...
  ' map elements.']);
end

PatchSharedWeights = false; % The same weights are used across all the H' x W' patches
if isempty(weights)
  PatchSharedWeights = true;
end

FMapSharedWeights = false; % The same weights are used across all the M 
% transform coefficients (feature map) of each patch.
if numel(g) == opts.Nbrs
  FMapSharedWeights = true;
end

if (isempty(dzdy) || nargin < 7)    
  useGPU = isa(x,'gpuArray');
  % Number of rows and columns of valid patches
  patchDims = [size(x,1),size(x,2)];
  NI = size(x,4);
  
  if (size(idx,1) ~= patchDims(1) || size(idx,2) ~= patchDims(2) || ...
      size(idx,3) ~= opts.Nbrs || size(idx,4) ~= NI)
    error('nn_pcgft:: idx does not have the correct dimensions.');
  end
  
  if PatchSharedWeights % We use common weights for each spatial dimension
    weights = g;
  else
    if FMapSharedWeights % We use common weights for each feature map element
      weights = bsxfun(@times,weights,reshape(g,[1, 1, opts.Nbrs, 1]));
    else
      Wr = size(weights,1); Wc = size(weights,2); WN = size(weights,4);
      weights = reshape(weights,[],1,opts.Nbrs,WN); 
      weights = bsxfun(@times,weights,reshape(g,[1, numFilters, opts.Nbrs, 1]));
      weights = reshape(weights,Wr,Wc,numFilters*opts.Nbrs,WN);
    end
  end
  [idx, n, I] = misc.FMapNLSumT_helper(idx);
  
  if opts.conserveMemory   
    
    if useGPU
      y = FMapNLSumT_gpu(x,weights,idx-1,n,I-1);
    else
      y = FMapNLSumT_cpu(x,weights,idx-1,n,I-1);
    end
    
    % Spatial transpose convolution - y is of size H x W x 1 x NI
    y = nn_convt_frn(y,h,wh,[],'stride',opts.stride,'cuDNN',opts.cuDNN,...
      'conserveMemory',true);
    
  else
    
    y = cell(2,1);
    
    if useGPU
      y{1} = FMapNLSumT_gpu(x,weights,idx-1,n,I-1);
    else
      y{1} = FMapNLSumT_cpu(x,weights,idx-1,n,I-1);
    end
    
    % Spatial transpose convolution - y{2}{2} is of size H x W x 1 x NI
    y{2} = nn_convt_frn(y{1},h,wh,[],'stride', opts.stride, 'cuDNN', ...
      opts.cuDNN);
    
    y{1} = {y{2}{1};y{1}};               
    y{2} = y{2}{2};    
  end    
else
  
  useGPU = isa(dzdy,'gpuArray');
  patchDims = [size(x{1}{2},1),size(x{1}{2},2)];
  NI = size(dzdy,4);
    
  if (size(idx,1)~=patchDims(1) || size(idx,2)~=patchDims(2) || ...
      size(idx,3)~=opts.Nbrs || size(idx,4)~=NI)
    error('nn_pgcft:: idx does not have the correct dimensions.');
  end
  
  [dzdy,dh] = nn_convt_frn(x{1},h,wh,dzdy,'cuDNN',opts.cuDNN,'derParams', ...
    opts.derParams(1),'stride',opts.stride);  
  x{1} = [];
  
  if opts.derParams(2)
    [widx,n,I] = misc.DerFMapNLSumT_helper(idx);
    dg = zeros(size(g),'like',g);

    if useGPU 
      DerFMapNLSumT = @DerFMapNLSumT_gpu; 
    else
      DerFMapNLSumT = @DerFMapNLSumT_cpu;
    end

    if FMapSharedWeights && PatchSharedWeights
      for k=1:opts.Nbrs
        tmp = DerFMapNLSumT(x{2},dzdy,widx(:,k,:)-1,n(:,k,:),...
          I(:,k,:)-1);
        dg(k) = sum(reshape(tmp,[],1));
      end
    elseif FMapSharedWeights && ~PatchSharedWeights
      for k=1:opts.Nbrs
        tmp = DerFMapNLSumT(bsxfun(@times,x{2},weights(:,:,k,:)),...
          dzdy,widx(:,k,:)-1,n(:,k,:),I(:,k,:)-1);
        dg(k) = sum(reshape(tmp,[],1));
      end
    elseif ~FMapSharedWeights && PatchSharedWeights
      for k=1:opts.Nbrs        
        tmp = DerFMapNLSumT(x{2},dzdy,widx(:,k,:)-1,n(:,k,:),...
          I(:,k,:)-1);
        sind = (k-1)*numFilters+1:numFilters*k;
        dg(sind) = sum(sum(reshape(tmp,[],numFilters,NI),1),3);
      end
    else
      for k=1:opts.Nbrs
        tmp = DerFMapNLSumT(bsxfun(@times,x{2},weights(:,:,k,:)),...
          dzdy,widx(:,k,:)-1,n(:,k,:),I(:,k,:)-1);        
        sind = (k-1)*numFilters+1:numFilters*k;
        dg(sind) = sum(sum(reshape(tmp,[],numFilters,NI),1),3);
      end
    end
  else
    dg = [];
  end
  
  if PatchSharedWeights % We use common weights for each spatial dimension
    weights = g;
  else
    if FMapSharedWeights % We use common weights for each feature map element
      weights = bsxfun(@times,weights,reshape(g,[1, 1, opts.Nbrs, 1]));
    else
      Wr = size(weights,1); Wc = size(weights,2); WN = size(weights,4);
      weights = reshape(weights,[],1,opts.Nbrs,WN); 
      weights = bsxfun(@times,weights,reshape(g,[1, numFilters, opts.Nbrs, 1]));
      weights = reshape(weights,Wr,Wc,numFilters*opts.Nbrs,WN);
    end
  end  
  
  if useGPU
    y = FMapNLSum_gpu(dzdy,weights,idx-1);
  else
    y = FMapNLSum_cpu(dzdy,weights,idx-1);
  end
  
end

% if FMapSharedWeights
%   for k=1:opts.Nbrs
%     if PatchSharedWeights
%       tmp = DerFMapNLSumT(x{2},dzdy,widx(:,k,:)-1,n(:,k,:),...
%         I(:,k,:)-1);
%     else
%       tmp = DerFMapNLSumT(bsxfun(@times,x{2},weights(:,:,k,:)),...
%         dzdy,widx(:,k,:)-1,n(:,k,:),I(:,k,:)-1);
%     end
%     dg(k) = sum(reshape(tmp,[],1));
%   end
% else
%   for k=1:opts.Nbrs
%     if PatchSharedWeights
%       tmp = DerFMapNLSumT(x{2},dzdy,widx(:,k,:)-1,n(:,k,:),...
%         I(:,k,:)-1);
%     else
%       tmp = DerFMapNLSumT(bsxfun(@times,x{2},weights(:,:,k,:)),...
%         dzdy,widx(:,k,:)-1,n(:,k,:),I(:,k,:)-1);
%     end
%     sind = (k-1)*numFilters+1:numFilters*k;
%     dg(sind) = sum(sum(reshape(tmp,[],numFilters,NI),1),3);
%   end
% end


