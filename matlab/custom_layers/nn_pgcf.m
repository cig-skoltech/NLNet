function [y,dh,dg] = nn_pgcf(x,h,wh,weights,g,idx,dzdy,varargin)

% NN_PGCF (Patch Group Collaborative Filtering)
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
%   X is an array of dimension H x W x 1 x N where (H,W) are
%   the height and width of the image stack and N the number of images in 
%   the stack.
%
%   H is an array of dimension PH x PW x 1 x K where (PH,PW)
%   are the filter height and width and K the number of filters in the
%   bank. 
%
%   WH is an array of size M x K (used in VL_NNCONV_FRN)
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
%   [DZDX,DH,DG] = NN_PGCF(X, H, WH, G, idx, DZDY) 
%   computes the derivatives of the block projected onto DZDY. DZDX, DH 
%   and DG have the same dimensions as X, WH and G, respectively. In 
%   the backward mode X is 2x1 cell array where X{1}=Y{2} and X{2}=Y{1}.
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

% % Parameters that reproduce the steps of nn_pgcf.
%
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

if numel(opts.derParams) <  2
  opts.derParams = [opts.derParams,repmat(opts.derParams(1),1,...
    2-numel(opts.derParams))];
end

patchSize = [size(h,1),size(h,2)];
numFilters = size(wh,1);

assert(size(wh,2) == size(h,4), ['nn_pgcf:: Invalid input for ' ...
  'wh - dimensions mismatch.']);

assert(isempty(weights) || all(size(weights) == size(idx)), ...
  'nn_pgcf:: Invalid input for weights - dimensions mismatch.');

if numel(opts.stride) == 1
  opts.stride = opts.stride*[1,1];
end

if numel(g) ~= opts.Nbrs && numel(g) ~= opts.Nbrs*numFilters
  error(['nn_pgcf:: The number of the neighbor-filter coefficients ' ...
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
  [Nx,Ny,Nc,NI] = size(x);
  useGPU = isa(x,'gpuArray');
  
  if Nc ~= 1
    error('nn_pgcf:: Only grayscale images are accepted as inputs.');
  end  
  % Number of rows and columns of valid patches
  patchDims = max(floor(([Nx,Ny]-patchSize)./opts.stride)+1,0); %[H',W']
  % Number of spatial filters
  
  if (size(idx,1) ~= patchDims(1) || size(idx,2) ~= patchDims(2) || ...
      size(idx,3) ~= opts.Nbrs || size(idx,4) ~= NI)
    error('nn_pcgf:: idx does not have the correct dimensions.');
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
  
  if opts.conserveMemory   
    % Spatial convolution - y is of size H' x W' x numFilters x NI
    y = nn_conv_frn(x,h,wh,[],'stride',opts.stride,'cuDNN',opts.cuDNN,...
      'conserveMemory',true);
    
    if useGPU
      y = FMapNLSum_gpu(y,weights,idx-1);
    else
      y = FMapNLSum_cpu(y,weights,idx-1);
    end
    
  else
    
    % Spatial convolution - y{2} is of size H' x W' x numFilters x NI
    y = nn_conv_frn(x,h,wh,[],'stride', opts.stride, 'cuDNN', ...
      opts.cuDNN);
    
    y{1} = {y{1},[size(x), ones(1,4-ndims(x))]};                      
    
    if useGPU
      y{3} = FMapNLSum_gpu(y{2},weights,idx-1);
    else
      y{3} = FMapNLSum_cpu(y{2},weights,idx-1);
    end
    
  end    
else
  
  patchDims = size(x{1}(:,:,1,1));
  if numel(x{2}{2}) == 4, NI = x{2}{2}(4); else; NI = 1; end
  
  if (size(idx,1)~=patchDims(1) || size(idx,2)~=patchDims(2) || ...
      size(idx,3)~=opts.Nbrs || size(idx,4)~=NI)
    error('nn_pgcf:: idx does not have the correct dimensions.');
  end
  
  if opts.derParams(2)
    dg = zeros(size(g),'like',g);
    
    for k = 1:opts.Nbrs
      ind = repmat(reshape(idx(:,:,k,:),[prod(patchDims) 1 NI]),[1,numFilters,1]);
      ind = bsxfun(@plus,ind,reshape(uint32(0:numFilters-1)*prod(patchDims),[1,numFilters,1]));
      ind = bsxfun(@plus,ind,reshape(uint32(0:NI-1)*prod(patchDims)*numFilters,[1 1 NI]));
      
      if PatchSharedWeights && FMapSharedWeights
        dg(k) = sum(reshape(x{1}(ind),[],1).*reshape(dzdy,[],1));
      elseif PatchSharedWeights && ~FMapSharedWeights
        sind = (k-1)*numFilters+1:numFilters*k;
        dg(sind) = sum(sum(reshape(x{1}(ind),[],numFilters,NI).* ...
          reshape(dzdy,[],numFilters,NI),1),3);
      elseif ~PatchSharedWeights && FMapSharedWeights
        dg(k) = sum(reshape(x{1}(ind),[],1).* ...
          reshape(bsxfun(@times,dzdy,weights(:,:,k,:)),[],1));
      else
        sind = (k-1)*numFilters+1:numFilters*k;
        dg(sind) = sum(sum(reshape(x{1}(ind),[],numFilters,NI).* ...
          reshape(bsxfun(@times,dzdy,weights(:,:,k,:)),[],numFilters,NI),1),3);
      end      
    end
  else
    dg = [];
  end
  x{1} =[];
  
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
  if isa(dzdy,'gpuArray')
    % We use idx-1 and I-1 instead of idx and I because this is a mex
    % file and therefore the indexing starts from zero and not 1.
    y = FMapNLSumT_gpu(dzdy,weights,idx-1,n,I-1);
  else
    y = FMapNLSumT_cpu(dzdy,weights,idx-1,n,I-1);
  end    
  
  [y,dh] = nn_conv_frn(x{2},h,wh,y,'cuDNN',opts.cuDNN,'derParams', ...
    opts.derParams(1),'stride',opts.stride);    
 
end


%   if opts.derParams(2)
%     dg = zeros(size(g),'like',g);        
%     if PatchSharedWeights
%       for k = 1:opts.Nbrs
%         ind = repmat(reshape(idx(:,:,k,:),[prod(patchDims) 1 NI]),[1,numFilters,1]);
%         ind = bsxfun(@plus,ind,reshape(uint32(0:numFilters-1)*prod(patchDims),[1,numFilters,1]));
%         ind = bsxfun(@plus,ind,reshape(uint32(0:NI-1)*prod(patchDims)*numFilters,[1 1 NI]));
%         if FMapSharedWeights
%           dg(k) = sum(reshape(x{1}(ind),[],1).*reshape(dzdy,[],1));
%         else
%           sind = (k-1)*numFilters+1:numFilters*k;
%           dg(sind) = sum(sum(reshape(x{1}(ind),[],numFilters,NI).* ...
%             reshape(dzdy,[],numFilters,NI),1),3);
%         end
%       end
%     else
%       for k = 1:opts.Nbrs
%         ind = repmat(reshape(idx(:,:,k,:),[prod(patchDims) 1 NI]),[1,numFilters,1]);
%         ind = bsxfun(@plus,ind,reshape(uint32(0:numFilters-1)*prod(patchDims),[1,numFilters,1]));
%         ind = bsxfun(@plus,ind,reshape(uint32(0:NI-1)*prod(patchDims)*numFilters,[1 1 NI]));
%         if FMapSharedWeights
%           dg(k) = sum(reshape(x{1}(ind),[],1).* ...
%             reshape(bsxfun(@times,dzdy,weights(:,:,k,:)),[],1));
%         else
%           sind = (k-1)*numFilters+1:numFilters*k;
%           dg(sind) = sum(sum(reshape(x{1}(ind),[],numFilters,NI).* ...
%             reshape(bsxfun(@times,dzdy,weights(:,:,k,:)),[],numFilters,NI),1),3);
%         end
%       end
%     end
%     clear sind;
%   else
%     dg = [];
%   end
%   x{1} = [];



