function y = LumChrom2rgb(x,dzdy,varargin)
opts.cuDNN = 'cuDNN';
opts.Op = [1/3 1/3 1/3; 0.5 0 -0.5; 0.25 -0.5 0.25]';
opts.scale = 1;
opts = vl_argparse(opts, varargin);

if nargin < 2
  dzdy = [];
end

invOp = [];
if isa(opts.Op,'char')
  switch opts.Op
    case 'opp'
      opts.Op = [1/3 1/3 1/3; 0.5  0  -0.5; 0.25  -0.5  0.25]';
      invOp = [1 1 2/3;1 0 -4/3;1 -1 2/3]';
    case 'yCbCr'
      opts.Op = [0.299, 0.587, 0.114;-0.16873660714285, -0.33126339285715, 0.5;0.5, -0.4186875, -0.0813125]';
      invOp = inv(opts.Op);
    case 'none'
      return;
    otherwise
      error('LumChrom2rgb:: Unknown color transformation.');
  end
end

if isempty(opts.Op)
  opts.Op = [1/3 1/3 1/3; 0.5  0  -0.5; 0.25  -0.5  0.25]';
end

if isempty(invOp)
  invOp = inv(opts.Op);
end


Nc = size(parseInputs(x,dzdy),3);
cid = classType(parseInputs(x,dzdy));
useGPU = isa(parseInputs(x,dzdy),'gpuArray');

if Nc ~= 3
  error('LumChrom2rgb:: input image must have three channels.');
end

if useGPU
  opts.Op = reshape(gpuArray(cast(opts.Op,cid)),1,1,3,3);
  invOp = reshape(gpuArray(cast(invOp,cid)),1,1,3,3);
else
  opts.Op = reshape(cast(opts.Op,cid),1,1,3,3);
  invOp = reshape(cast(invOp,cid),1,1,3,3);
end

maxV = sum(max(squeeze(opts.Op),0),1);
minV = sum(min(squeeze(opts.Op),0),1);

if isempty(dzdy)  
  x = bsxfun(@times,x,reshape((maxV-minV),1,1,3));
  x = bsxfun(@plus,x,reshape(minV,1,1,3));
  
  y = vl_nnconv(x,invOp,[],opts.cuDNN)*opts.scale;
else
  y = vl_nnconv(zeros(size(dzdy),'like',dzdy),invOp,[],dzdy,...
    opts.cuDNN,'NoDerBiases','NoDerFilters');
  y = bsxfun(@times,y,reshape(opts.scale*(maxV-minV),1,1,3));
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
