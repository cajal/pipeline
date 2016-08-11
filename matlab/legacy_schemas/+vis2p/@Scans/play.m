function play( obj, frames,trace_opt )

keys = fetch(obj);
for ikey = length( keys )
    key = keys(ikey);
    disp('Playing movie for');
    disp(key);
    tpr = tpReader(Scans(key));
    if strcmp(fetch1(Scans(key),'scan_prog'),'MPScan')
        
        gChan = getAlignedImagingChannel( tpr, 1 );
        rChan = getAlignedImagingChannel( tpr, 2 );
    else
        movies = tpr.read([1 2]);
        gChan = movies(:,:,:,1);
        rChan = movies(:,:,:,2);
        clear movies
    end
    sz = size(gChan);
    
    if nargin<=1 || isempty(frames)
        frames=1:sz(3);
    end
    
    if nargin>2
        traces = fetchn(Traces(['trace_opt =' num2str(trace_opt)]).*obj,'trace');
        traces = cell2mat(traces');
    end
    % play
    
    for iFrame = 1:10:length(frames) -1000
        clf
        %       subplot(1,2,1);
        
        %         img = anscombe( cat( 3, rChan(:,:,frames(iFrame))/2, gChan(:,:,frames(iFrame)), zeros(sz(1:2)) ) );
        %         im = imshow( img );
        
        imagesc(gChan(:,:,frames(iFrame)));
        colormap gray
        axis image
        %        subplot(1,2,2);
        %
        %          plot(bsxfun(@plus,traces(iFrame:iFrame+1000,:),(1:size(traces,2))/4))
        drawnow;
    end
end

function img = anscombe( img )
img = sqrt(max(0,double(img)+4))/sqrt(2052);