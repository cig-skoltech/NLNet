function [y,dzdw] = nn_conv_frn(x,f,w,dzdy,varargin)
%   CNN Convolution with filter response normalization
%   Y = NN_CONV_FRN(X, F, W) computes the convolution of the image X
%   with the filter bank F. If ConserveMemory is set to false then Y is
%   a 2 x 1 cell array where Y{2} is the output of the layer.
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
%   [DX, DW] = NN_CONV_FRN(X, F, W, DY) computes the derivatives of
%   the operator projected onto P. DX, DW and DY have the same
%   dimensions as X, W, and Y, respectively. In the backward mode X 
%   is a cell array 2x1 where X{1} = Y{1} and X{2} is either the input X
%   or a vector with the size of X. If w is an empty array then X is just
%   the input of the forward mode.
%
%
%   NN_CONV_FRN(..., 'option', value, ...) accepts the following
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

if frn_flag && (size(w,2) ~= size(f,4))
  error('nn_conv_frn :: Invalid input w - dimensions mismatch.');
end

if nargin < 4 || isempty(dzdy)
  dzdw = [];
  if opts.conserveMemory || ~frn_flag
    if frn_flag
      sz = size(f);
      f = reshape(nn_filtResNorm(reshape(f,sz(1),sz(2)*sz(3),sz(4)),w,[],...
        'cuDNN',opts.cuDNN),[sz(1),sz(2),sz(3),size(w,1)]);
    end    
    y = vl_nnconv(x,f,[],'stride',opts.stride,opts.cuDNN);    
  else
    y = cell(2,1);
    y{1} = vl_nnconv(x,f,[],'stride',opts.stride,opts.cuDNN);
    y{2} = nn_filtResNorm(y{1},w,[],'cuDNN',opts.cuDNN);
  end  
  
%   if opts.conserveMemory || ~frn_flag
%     y = vl_nnconv(x,f,[],'stride',opts.stride,opts.cuDNN);
%     if frn_flag
%       y =  nn_filtResNorm(y,w,[],'cuDNN',opts.cuDNN);
%     end
%   else
%     y = cell(2,1);
%     y{1} = vl_nnconv(x,f,[],'stride',opts.stride,opts.cuDNN);
%     y{2} =  nn_filtResNorm(y{1},w,[],'cuDNN',opts.cuDNN);
%   end
  
else
  if frn_flag
    [y,dzdw] = nn_filtResNorm(x{1},w,dzdy,'derParams',opts.derParams,'cuDNN',opts.cuDNN);
    x{1} = [];
    if isvector(x{2})
      x{2} = [x{2}(:)' ones(1,4-numel(x{2}))];
      x{2} = zeros(x{2},'like',y);
    end
    y = vl_nnconv(x{2},f,[],y,'stride', ...
      opts.stride,opts.cuDNN,'NoDerFilters','NoDerBiases');
  else
    y = vl_nnconv(x,f,[],dzdy,'stride',opts.stride,opts.cuDNN,...
      'NoDerFilters','NoDerBiases'); 
  end
end
  
  
  
  
  
  
