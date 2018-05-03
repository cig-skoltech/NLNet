function nlnet_BSDS_validation_results(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
             'matlab', 'vl_layers', 'vl_setupnn.m')) ;

opts.patchSize = [5 5];
opts.color = false;
opts.train_mode = 'greedy';
opts.noise_std = 15;
opts.gpus = [];
opts.imdbPath = 'BSDS500/';
opts.modelsPath = 'network/models/';
opts.savePath = 'network/AvgResults/';
opts.fileList = 'network/BSDS_validation_list.txt';

opts = vl_argparse(opts,varargin);

if ~opts.color
  net_name = ['nlnet_' num2str(opts.patchSize(1)) 'x' num2str(opts.patchSize(2)) ...
    '_std=' num2str(opts.noise_std) '_' opts.train_mode '.mat'];
  res_name = ['res_' net_name];
else
  net_name = ['cnlnet_' num2str(opts.patchSize(1)) 'x' num2str(opts.patchSize(2)) ...
    '_std=' num2str(opts.noise_std) '_' opts.train_mode '.mat'];
  res_name = ['res_' net_name];
end

load(fullfile(opts.modelsPath,net_name)); 

run_denoise_nlnet(net,opts.fileList,'color',opts.color,'noise_std', ...
  opts.noise_std,'gpus',opts.gpus,'imdbPath',opts.imdbPath,'savePath',...
  fullfile(opts.savePath,res_name));


function [psnr_measure,xe_tall,xe_fat,y_tall,y_fat,x_tall,x_fat]= run_denoise_nlnet(net,fileList,varargin)
opts.imdbPath = '../BSDS500/';
opts.color = false;
opts.noise_std = 25;
opts.gpus = [];
opts.savePath = [];
opts.randn_seed = 19092015;
opts.batchSize = 20;

opts = vl_argparse(opts,varargin);

if opts.color
  opts.imdbPath = fullfile(opts.imdbPath,'color');
else
  opts.imdbPath = fullfile(opts.imdbPath,'gray');
end

fileID = fopen(fileList,'r');
C = strsplit(fscanf(fileID,'%s'),'.jpg');
C(end) = [];
fclose(fileID);

x_tall = single([]); % 481 x 321  images
x_fat = single([]); % 321 x 481 images

ctr_tall = 1;
ctr_fat = 1;
for k = 1:numel(C)
  f = single(imread([opts.imdbPath filesep C{k} '.jpg']));
  if size(f,1) > size(f,2)
    x_tall(:,:,:,ctr_tall) = f;
    ctr_tall = ctr_tall + 1;
  else
    x_fat(:,:,:,ctr_fat) = f;
    ctr_fat = ctr_fat + 1;
  end
end

% Initialize the seed for the random generator
s = RandStream('mt19937ar','Seed',opts.randn_seed);
RandStream.setGlobalStream(s);

% The degraded input that we feed to the network and we want to
% reconstruct.
y_tall = x_tall + opts.noise_std * randn(size(x_tall),'like',x_tall);
y_fat = x_fat + opts.noise_std * randn(size(x_fat),'like',x_fat);


% Run the NLNet network

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
  [x_tall,x_fat,y_tall,y_fat] = misc.move_data('gpu',x_tall,x_fat,y_tall,y_fat);
  net = net_move(net,'gpu');
end

if opts.color
  idx = misc.patchMatch(nn_pad(sum(y_tall/255,3)/3,net.layers{2}.padSize),'stride',net.layers{2}.stride,'Nbrs',net.layers{2}.Nbrs,'searchwin',[15 15],'patchsize',size(net.layers{2}.filters{1}(:,:,1,1)));
  Params = net.meta.netParams;
  N = size(y_tall,4);
  for t = 1:opts.batchSize:size(y_tall,4)
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1,N);
    Params.Nbrs_idx = idx(:,:,:,batchStart:batchEnd);
    res = cnlcf_eval(net,y_tall(:,:,:,batchStart:batchEnd),[],[],...
      'conserveMemory',true,'netParams',Params);
    xe_tall(:,:,:,batchStart:batchEnd) = res(end).x;
    clear res;
  end
  
  idx = misc.patchMatch(nn_pad(sum(y_fat/255,3)/3,net.layers{2}.padSize),'stride',net.layers{2}.stride,'Nbrs',net.layers{2}.Nbrs,'searchwin',[15 15],'patchsize',size(net.layers{2}.filters{1}(:,:,1,1)));
  N = size(y_fat,4);
  for t = 1:opts.batchSize:size(y_fat,4)
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1,N);
    Params.Nbrs_idx = idx(:,:,:,batchStart:batchEnd);
    res = cnlcf_eval(net,y_fat(:,:,:,batchStart:batchEnd),[],[],...
      'conserveMemory',true,'netParams',Params);
    xe_fat(:,:,:,batchStart:batchEnd) = res(end).x;
    clear res;
  end
  
else
  idx = misc.patchMatch(nn_pad(y_tall,net.layers{1}.padSize),'stride',net.layers{1}.stride,'Nbrs',net.layers{1}.Nbrs,'searchwin',[15 15],'patchsize',size(net.layers{1}.filters{1}(:,:,1,1)));  
  Params = net.meta.netParams;
  N = size(y_tall,4);
  for t = 1:opts.batchSize:size(y_tall,4)
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1,N);
    Params.Nbrs_idx = idx(:,:,:,batchStart:batchEnd);
    res = nlcf_eval(net,y_tall(:,:,:,batchStart:batchEnd),[],[],...
      'conserveMemory',true,'netParams',Params);
    xe_tall(:,:,:,batchStart:batchEnd) = res(end).x;
    clear res;
  end
  
  idx = misc.patchMatch(nn_pad(y_fat,net.layers{1}.padSize),'stride',net.layers{1}.stride,'Nbrs',net.layers{1}.Nbrs,'searchwin',[15 15],'patchsize',size(net.layers{1}.filters{1}(:,:,1,1)));  
  N = size(y_fat,4);
  for t = 1:opts.batchSize:size(y_fat,4)
    batchStart = t;
    batchEnd = min(t+opts.batchSize-1,N);
    Params.Nbrs_idx = idx(:,:,:,batchStart:batchEnd);
    res = nlcf_eval(net,y_fat(:,:,:,batchStart:batchEnd),[],[],...
      'conserveMemory',true,'netParams',Params);
    xe_fat(:,:,:,batchStart:batchEnd) = res(end).x;
    clear res;
  end
  
end

psnr_measure = zeros(numel(C),1,'like',x_tall);
ctr = 1;
for k = 1:size(x_fat,4)
  psnr_measure(ctr) = misc.psnr(xe_fat(:,:,:,k),x_fat(:,:,:,k),255);
  ctr = ctr+1;
end

for k = 1:size(x_tall,4)
  psnr_measure(ctr) = misc.psnr(xe_tall(:,:,:,k),x_tall(:,:,:,k),255);
  ctr = ctr+1;
end


if ~isempty(opts.gpus)
  [x_tall,x_fat,y_tall,y_fat,xe_tall,xe_fat,psnr_measure] = ...
    misc.move_data('cpu',x_tall,x_fat,y_tall,y_fat,xe_tall,xe_fat,psnr_measure);
end

if ~isempty(opts.savePath)
  save(opts.savePath,'x_tall','x_fat','y_tall','y_fat','xe_tall','xe_fat','psnr_measure');
end


