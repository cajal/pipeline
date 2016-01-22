%{
pre.Trace (imported) # calcium trace
-> pre.Segment
trace_id        : smallint               # mask number in segmentation
---
ca_trace                    : longblob                      # raw calcium trace
%}


% One-line plotting script:
% r = 'animal_id=5269';
% for k=fetch(pre.Segment*rf.Sync&r)',t=fetch1(rf.Sync&k,'frame_times');x=fetchn(pre.Trace&k,'ca_trace');x=[x{:}];plot(t-t(1),bsxfun(@plus,bsxfun(@rdivide,x,mean(x)),1:size(x,2)));keyboard;end

classdef Trace < dj.Relvar
    
    methods
        
        function plot(self)
            for key = fetch(pre.Segment & self)'
                X = fetchn(self & key, 'ca_trace');
                X = [X{:}];
                t = fetch1(pre.Sync & key, 'frame_times');
                X = bsxfun(@rdivide,X,mean(X));
                plot(t-t(1),X)
            end
        end
        
        
        function makeTuples(self, key)
            tic
            fixRaster = get_fix_raster_fun(pre.AlignRaster & key);
            fixMotion = get_fix_motion_fun(pre.AlignMotion & key);
            
            mask = fetch1(pre.Segment & key, 'mask');
            pixels = regionprops(mask, 'PixelIdxList');
            pixels = {pixels.PixelIdxList};
            ntraces = length(pixels);
            
            reader = pre.getReader(key, '~/cache');
            assert(reader.nslices == 1, 'deal with slices later')
            nframes = reader.nframes;
            traces = nan(nframes, ntraces, 'single');
            for iframe=1:nframes
                if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                    fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                end                
                frame = fixMotion(fixRaster(double(reader(:,:,:,:,iframe))), iframe);
                traces(iframe, :) = cellfun(@(pixels) mean(frame(pixels)), pixels);
            end
            
            % save
            for itrace=1:ntraces
                key.trace_id = itrace;
                key.ca_trace = traces(:,itrace);
                self.insert(key)
            end
            
        end
    end
end
