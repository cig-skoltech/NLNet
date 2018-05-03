function y = rgb2LumChrom(x,dzdy,varargin)
% Input x is assumed to be scaled in the range [0 1];
opts.cuDNN = 'cuDNN';
opts.Op = [1/3 1/3 1/3; 0.5 0 -0.5; 0.25 -0.5 0.25]';
opts.scale = 1;
opts = vl_argparse(opts, varargin);

if nargin < 2
  dzdy = [];
end

if isa(opts.Op,'char')
  switch opts.Op
    case 'opp'
      opts.Op = [1/3 1/3 1/3; 0.5  0  -0.5; 0.25  -0.5  0.25]';
    case 'yCbCr'
      opts.Op = [0.299, 0.587, 0.114;-0.16873660714285, -0.33126339285715, 0.5;0.5, -0.4186875, -0.0813125]';
    case 'none'
      return;
    otherwise
      error('rgb2LumChrom:: Unknown color transformation.');
  end
end

if isempty(opts.Op)
  opts.Op = [1/3 1/3 1/3; 0.5  0  -0.5; 0.25  -0.5  0.25]';
end


Nc = size(parseInputs(x,dzdy),3);
useGPU = isa(parseInputs(x,dzdy),'gpuArray');
cid = classType(parseInputs(x,dzdy));

if Nc ~= 3
  error('rgb2LumChrom:: input image must have three channels.');
end

if useGPU
  opts.Op = reshape(gpuArray(cast(opts.Op,cid)),1,1,3,3);
else
  opts.Op = reshape(cast(opts.Op,cid),1,1,3,3);
end

maxV = sum(max(squeeze(opts.Op),0),1);
minV = sum(min(squeeze(opts.Op),0),1);

if isempty(dzdy)
  y = vl_nnconv(x/opts.scale,opts.Op,[],opts.cuDNN);
  y = bsxfun(@minus,y,reshape(minV./(maxV-minV),1,1,3));
else
  y = vl_nnconv(zeros(size(dzdy),'like',dzdy),opts.Op,[],dzdy,...
    opts.cuDNN,'NoDerBiases','NoDerFilters')/opts.scale;
end


function cid = classType(x)

if isa(x,'gpuArray')
  cid = classUnderlying(x);
else
  cid = class(x);
end

function x = parseInputs(x,y)

if isempty(x)
  x = y;
end