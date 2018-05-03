function y = nn_pad(x, padSize , dzdy)
%NN_PAD CNN padding.
%   Y = NN_PAD(X, PADSIZE) pads symmetrically the spatial dimensions of
%   the input X.
%   PADSIZE specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
%
%   DZDX = NN_PAD(X, PADSIZE, DZDY) computes the derivative DZDX of the
%   function projected on the output derivative DZDY. DZDX has the same
%   dimension as X and DZDY the same dimension as Y.
%
%   DZDX = NN_PAD([], PADSIZE, DZDY) is an alternative to
%   the previous call in which X is omitted.

% stamatis@math.ucla.edu, 06/01/2015


if ~isempty(x)
  sz=size(x);
elseif ~isempty(dzdy)
  sz=size(dzdy);
  sz(1)=sz(1)-padSize(1)-padSize(2);
  sz(2)=sz(2)-padSize(3)-padSize(4);
else
  sz=[];
end

if numel(padSize)==1
  padSize=padSize*ones(1,4);
end

if padSize(1) > sz(1) || padSize(2) > sz(1) || padSize(3) > sz(2) || padSize(4) > sz(2)
  error('nn_pad:InvalidInput','padSize cannot be greater than inputSize.');
end

if nargin <= 2 || isempty(dzdy)
  
  sflag=false; % Check for equal-size padding on TOP-BOTTOM and LEFT-RIGHT
  if padSize(1)==padSize(2) && padSize(3)==padSize(4)
    sflag=true;
  end
  
  if sflag
    y=padarray(x,[padSize(1),padSize(3)],'both','symmetric');
  else
    y = padarray(x,[padSize(1),padSize(3)],'pre','symmetric');
    y = padarray(y,[padSize(2),padSize(4)],'post','symmetric');
  end
else
  dzdy(padSize(1)+1:2*padSize(1),:,:,:)=...
    dzdy(padSize(1)+1:2*padSize(1),:,:,:)+dzdy(padSize(1):-1:1,:,:,:);
  
  dzdy(end-2*padSize(2)+1:end-padSize(2),:,:,:)=...
    dzdy(end-2*padSize(2)+1:end-padSize(2),:,:,:)...
    +dzdy(end:-1:end-padSize(2)+1,:,:,:);
  
  dzdy(:,padSize(3)+1:2*padSize(3),:,:)=...
    dzdy(:,padSize(3)+1:2*padSize(3),:,:)+dzdy(:,padSize(3):-1:1,:,:);
  
  dzdy(:,end-2*padSize(4)+1:end-padSize(4),:,:)=...
    dzdy(:,end-2*padSize(4)+1:end-padSize(4),:,:)...
    +dzdy(:,end:-1:end-padSize(4)+1,:,:);
  
  y=dzdy(padSize(1)+1:end-padSize(2),padSize(3)+1:end-padSize(4),:,:);
end
