function correctedStackTiff(key,filepath,direction)
% direction can be 'normal' or 'reverse'
if nargin <3
    direction='normal';
end

stk = stack.loadCorrectedStack(key);

switch direction
    case 'normal'
        ind = 1:size(stk,4);
    case 'reverse'
        ind = size(stk,4):-1:1;
    otherwise
        error('direction can be ''normal'' or ''reverse''.');
end

for i=ind
    for c=1:size(stk,3)
        imwrite(uint16(mean(stk(:,:,c,i,:),5)),filepath,'writemode','append')
    end
end
