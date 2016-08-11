function blks = getblocks(idx, tol_len, tol_gap)
%
% Returs a cell array of indices into idx that represent continguous blocks
% in idx that are logically true. 
%
% There are two exceptions:
% 1) If the gap between two blocks is less than tol_gap, then the gap is
%   included in one block
% 2) If a block is shorted than tol_len, the it is ignored. 
%
if size(idx,1) > size(idx,2), idx = idx'; end

if nargin < 3
    tol_gap = 5;
end


d = diff(idx);
b = find(d > 0)+1;
e =find(d<0);
if idx(end), e = [e,length(idx)]; end
if idx(1), b = [1,b]; end

blks = {};
n = length(e);
count = 1;
for i = 1:length(e)
    if (i ~= n) && (b(i+1) - e(i) -1 <= tol_gap)
            b(i+1) = b(i);
            continue
    end
    if e(i) - b(i) < tol_len - 1, continue; end
    blks{count} = b(i):e(i);
    count = count + 1;
end