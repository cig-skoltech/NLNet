function [y,dzdw] = nn_convt_frn(x,f,w,dzdy,varargin)
%   CNN Transpose Convolution with filter response normalization
%   Y = NN_CONVT_FRN(X, F, W) computes the transpose of the convolution 
%   of the image X with the filter bank F. If ConserveMemory is set to 
%   false then Y is a 2 x 1 cell array where Y{2} is the output of the layer.
%
%   X is an array of dimension H x W x C x N where (H,W) are the
%   height and width of the image stack, C is the number of feature
%   channels, and N is the number of images in the batch.
%
%   F is an array of dimension FW x FH x FC x K where (FH,FW) are the
%   filter height and width and K the number o filters in the bank. FC
%   is the number of feature channels in each filter and must match
%   the number of feature channels C in X. Alternatively, FC can
%   *divide* the C; in this case, filters are assumed to form G=C/FC
%   *groups* of equal size (where G must divide K). Each group of
%   filters works on a consecutive subset of feature channels of the
%   input array X.
%
%   [DX, DW] = NN_CONVT_FRN(X, F, W, DY) computes the derivatives of
%   the operator projected onto DY. DX, DW and DY have the same
%   dimensions as X, W, and Y, respectively. In the backward mode X 
%   is a cell array 2 x 1  where X{1} = Y{1} and X{2} is the input of the
%   forward mode. If w is an empty array then X is just the input of the 
%   forward mode or a vector with its size.
%
%
%   NN_CONVT_FRN(..., 'option', value, ...) accepts the following
%   options:
%
%   `Stride`:: 1
%     The output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical and
%     horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%     allows specifying different downsampling factors for each
%     direction.
%
%   'derParams' :: true
%     If derParams is set to false then in the backward step dzdw is not
%     computed.
%
%   `ConserveMemory`:: false
%       If set to true saves memory by not storing the intermediate results
%       from the different layers. If used in training then the value
%       should be set to false.
%
%   `cuDNN`:: {'CuDNN'} | 'NoCuDNN'
%     It indicates whether CuDNN will be used or not during the computation
%     of the convolutions.
%
%
opts.derParams = true;
opts.cuDNN = 'cuDNN';
opts.stride = [1 1];
opts.conserveMemory = false;
opts = vl_argparse(opts,varargin);

frn_flag = true;
if isempty(w) || nargin < 3
  frn_flag = false;
end

err_msg = 'nn_convt_frn :: Invalid input w - dimensions mismatch.';
if frn_flag && (size(w,2) ~= size(f,4)) 
  error(err_msg);
end

if nargin < 4 || isempty(dzdy)
  dzdw = [];
  if opts.conserveMemory || ~frn_flag
%     if frn_flag
%       assert(size(w,1) == size(x,3), err_msg);
%       assert(size(w,2) == size(f,4), err_msg);
%       x = nn_filtResNormt(x,w,[],'cuDNN',opts.cuDNN);
%     end
    if frn_flag
      assert(size(w,1) == size(x,3), err_msg);
      assert(size(w,2) == size(f,4), err_msg);
      sz = size(f);
      f = reshape(nn_filtResNorm(reshape(f,sz(1),sz(2)*sz(3),sz(4)),w,[],...
        'cuDNN',opts.cuDNN),sz(1),sz(2),sz(3),size(w,1));
    end
    y = vl_nnconvt(x,f,[],'upsample',opts.stride,opts.cuDNN);
  else
    y = cell(2,1);
    assert(size(w,1) == size(x,3), err_msg);
    assert(size(w,2) == size(f,4), err_msg);
    
    y{1} = nn_filtResNormt(x,w,[],'cuDNN',opts.cuDNN);
    y{2} = vl_nnconvt(y{1},f,[],'upsample',opts.stride,opts.cuDNN);
    % We only need to keep the size of the input in vl_nnconvt since in the
    % backward step we do not compute the derivatives w.r.t weights of the
    % filters.
    y{1} = [size(y{1}), ones(1,4-ndims(y{1}))];
  end
  
else
  if frn_flag   
    if isvector(x{1})
      x{1} = [x{1}(:)' ones(1,4-numel(x{1}))];
      x{1} = zeros(x{1},'like',dzdy);
    end
    assert(size(w,1) == size(x{2},3), err_msg);
    assert(size(w,2) == size(f,4), err_msg);   
    
    y = vl_nnconvt(x{1},f,[],dzdy,'upsample',opts.stride,opts.cuDNN,...
      'NoDerFilters','NoDerBiases');
    x{1} = [];
        
    [y,dzdw] = nn_filtResNormt(x{2},w,y,'derParams',opts.derParams,...
      'cuDNN',opts.cuDNN);
  else
    if isvector(x)
      x = [x(:)' ones(1,4-numel(x))];
      x = zeros(x,'like',dzdy);
    end    
    y = vl_nnconvt(x,f,dzdy,'upsample',opts.stride,opts.cuDNN,...
      'NoDerFilters','NoDerBiases');
  end
end
  
  
  
  
  
  
