function [psnr_measure,x,y,f] = nlnet_denoise_demo(varargin)

opts.filePath = 'BSDS500/gray/102061.jpg'; % Path for the ground-truth image.
opts.noise_std = 25; % Standard deviation of the noise degrading the image.
opts.randn_seed = 19092015; % Seed for the random generator.
opts = vl_argparse(opts,varargin);

f = single(imread(opts.filePath));

% Initialize the seed for the random generator
s = RandStream('mt19937ar','Seed',opts.randn_seed);
RandStream.setGlobalStream(s);

% The degraded input that we feed to the network and we want to
% reconstruct.
y = f + opts.noise_std * randn(size(f),'like',f);

if ndims(f) > 2 && size(f,3) == 3 %#ok<ISMAT>
  opts.color = true;
else
  opts.color = false;
end


stdn = [15 25 50];
d = abs(opts.noise_std-stdn);
[~,ind] = min(d);

if opts.color
  m = load(sprintf('network/models/cnlnet_5x5_std=%2.0d_joint.mat',stdn(ind)));
else
  m = load(sprintf('network/models/nlnet_7x7_std=%2.0d_joint.mat',stdn(ind)));
end
net = m.net;

% Run the NLNet network
opts.gpus = []; % To run the network on a gpu you need to set the gpu id,
% i.e., opts.gpus = 1;
if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
  [f,y] = misc.move_data('gpu',f,y);
  net = net_move(net,'gpu');
end

if opts.color
  idx = misc.patchMatch(nn_pad(sum(y/255,3)/3,net.layers{2}.padSize),'stride',net.layers{2}.stride,'Nbrs',net.layers{2}.Nbrs,'searchwin',[15 15],'patchsize',size(net.layers{2}.filters{1}(:,:,1,1)));
  Params = net.meta.netParams;
  Params.Nbrs_idx = idx;
  clear idx;
  res = cnlcf_eval(net,y,[],[],'conserveMemory',true,'netParams',Params);
  x = res(end).x;
  clear res;
else
  idx = misc.patchMatch(nn_pad(y,net.layers{1}.padSize),'stride',net.layers{1}.stride,'Nbrs',net.layers{1}.Nbrs,'searchwin',[15 15],'patchsize',size(net.layers{1}.filters{1}(:,:,1,1)));
  Params = net.meta.netParams;
  Params.Nbrs_idx = idx;
  clear idx;
  res = nlcf_eval(net,y,[],[],'conserveMemory',true,'netParams',Params);
  x = res(end).x;
  clear res;
end

psnr_measure =  misc.psnr(x,f,255);

