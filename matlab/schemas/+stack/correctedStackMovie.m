function correctedStackMovie(key,filepath,direction)
% direction can be 'normal' or 'reverse'
if nargin <3
    direction='normal';
end

stk = stack.loadCorrectedStack(key);

figure
writer = VideoWriter(filepath, 'MPEG-4');
writer.Quality = 100;
writer.FrameRate = 24;
writer.open
set(gcf,'color','w')

switch direction
    case 'normal'
        ind = 1:size(stk,4);
    case 'reverse'
        ind = size(stk,4):-1:1;
    otherwise
        error('direction can be ''normal'' or ''reverse''.');
end

for i=ind
    g = squeeze(mean(stk(:,:,1,i,:),5));
    g = g-min(g(:));
    g = g/quantile(g(:),.99);
    
    if size(stk,3)>1
        r = squeeze(mean(stk(:,:,2,i,:),5));
        r = r-min(r(:));
        r = r/quantile(r(:),.95);
    else
        r = ones(size(g))*.4;
    end
    
    g(g>1)=1; r(r>1)=1;
    image(cat(3,r,g,ones(size(g))*.4));
    axis image off
    drawnow
    
    writer.writeVideo(getframe(gcf));
end

writer.close
