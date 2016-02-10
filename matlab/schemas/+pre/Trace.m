%{
pre.Trace (imported) # calcium trace
-> pre.SegmentMask
---
ca_trace                    : longblob                      # raw calcium trace
%}


% One-line plotting script:
% r = 'animal_id=5269';
% for k=fetch(pre.Segment*rf.Sync&r)',t=fetch1(rf.Sync&k,'frame_times');x=fetchn(pre.Trace&k,'ca_trace');x=[x{:}];plot(t-t(1),bsxfun(@plus,bsxfun(@rdivide,x,mean(x)),1:size(x,2)));keyboard;end

classdef Trace < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = pre.Segment
    end
    methods
        
        function plot(self)
            for key = fetch(pre.Segment & self)'
                X = fetchn(self & key, 'ca_trace');
                X = [X{:}];
                t = fetch1(rf.Sync & key, 'frame_times');
                X = bsxfun(@plus,bsxfun(@rdivide,X,mean(X))/2,1:size(X,2));
                plot(t-t(1),X)
            end
        end
        
    end
    
    methods(Access = protected)
        
        function makeTuples(self, key)
            tic
            fixRaster = get_fix_raster_fun(pre.AlignRaster & key);
            fixMotion = get_fix_motion_fun(pre.AlignMotion & key);
            
            [pixels, weights, maskKeys] = fetchn(pre.SegmentMask & key, 'mask_pixels', 'mask_weights');
            ntraces = length(pixels);
            
            reader = pre.getReader(key, '~/cache');
            nframes = reader.nframes;
            traces = nan(nframes, ntraces, 'single');
            for iframe=1:nframes
                if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                    fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                end                
                frame = fixMotion(fixRaster(double(reader(:,:,1,key.slice,iframe))), iframe);
                traces(iframe, :) = cellfun(@(pixels,weights) mean(frame(pixels).*weights), pixels, weights);
            end
            
            % save
            for itrace=1:ntraces
                self.insert(setfield(maskKeys(itrace), 'ca_trace', traces(:, itrace))) %#ok<SFLD>
            end
            
        end
    end
end
