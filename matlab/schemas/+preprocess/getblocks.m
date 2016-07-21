function blks = getblocks(idx, tol_len, tol_gap)

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
    
    if i ~= n
        b(i+1) - e(i)
        if b(i+1) - e(i) -1 <= tol_gap
            
            b(i+1) = b(i);
            continue
        end
    end
    if e(i) - b(i) < tol_len - 1, continue; end
    blks{count} = b(i):e(i);
    count = count + 1;
end