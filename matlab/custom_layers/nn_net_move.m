function net = nn_net_move(net, destination)
%NN_NET_MOVE  Move a network between CPU and GPU.
%   NET = NN_NET_MOVE(NET, 'gpu') moves the network to the
%   current GPU device. NET = NN_NET_MOVE(NET, 'cpu') moves the
%   network to the CPU.
%
%   See also: VL_SIMPLENN().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown destination ''%s''.', destination) ;
end
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv', 'convt', 'bnorm', 'conv_sb', 'convt_sb', 'shrink', 'l2proj'}
      for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
        c = char(f) ;
        if isfield(net.layers{l}, c)
          net.layers{l}.(c) = moveop(net.layers{l}.(c)) ;
        end
      end
      for f = {'weights', 'momentum'}
        c = char(f) ;
        if isfield(net.layers{l}, c)
          for j=1:numel(net.layers{l}.(c))
            net.layers{l}.(c){j} = moveop(net.layers{l}.(c){j}) ;
          end
        end
      end
      
      if isfield(net.layers{l}, 'rbf_means')
        net.layers{l}.rbf_means = moveop(net.layers{l}.rbf_means);
      end     
    otherwise
      % nothing to do ?
  end
end
