%{
rf.Trace (imported) # calcium trace
-> rf.Segment
trace_id        : smallint               # mask number in segmentation
---
ca_trace                    : longblob                      # raw calcium trace
%}


% One-line plotting script:
% r = 'animal_id=5269';
% for k=fetch(rf.Segment*rf.Sync&r)',t=fetch1(rf.Sync&k,'frame_times');x=fetchn(rf.Trace&k,'ca_trace');x=[x{:}];plot(t-t(1),bsxfun(@plus,bsxfun(@rdivide,x,mean(x)),1:size(x,2)));keyboard;end

classdef Trace < dj.Relvar 

    methods
        
        function plot(self)
            for key = fetch(rf.Segment & self)'
                X = fetchn(self & key, 'ca_trace');
                X = [X{:}];
                t = fetch1(rf.Sync & key, 'frame_times');
                X = bsxfun(@rdivide,X,mean(X));
                plot(t-t(1),X)
            end
        end
        
        function makeTuples(self, key)
            
            reader = rf.getReader(key);
            [masks,sliceKeys] = fetchn(rf.Segment & key, 'mask');
            xymotion = fetch1(rf.Align & key, 'motion_xy');
            
            nSlices = length(sliceKeys);
            assert(reader.nSlices == nSlices)
            
            assert(~any(cellfun(@islogical, masks)), 'mask region must be labeled')
            
            % extract pixels for each trace
            pixels = ...
                cellfun(@(mask) arrayfun(@(x) x.PixelIdxList, ...
                regionprops(mask,'PixelIdxList'), 'uni', false), ...
                masks, 'uni', false);
            
            disp 'loading traces...'
            traces = cell(nSlices, 1);
            blockSize = 300;
            [rasterPhase, fillFraction] = fetch1(rf.Align & key, ...
                'raster_phase', 'fill_fraction');
            
            while ~reader.done
                if strcmp(fetch1(rf.Session & key,'fluorophore'),'TN-XXL')
                    % ratiometric traces
                    block=reader.read([1 2], 1:reader.nSlices, blockSize); 
                    xy = xymotion(:,:,1:size(block.channel1,4));
                    xymotion(:,:,1:size(block.channel1,4)) = [];
                    if strcmp(reader.hdr.scanMode, 'bidirectional')
                        block.channel1 = ne7.ip.correctRaster(block.channel1, rasterPhase, fillFraction);
                        block.channel2 = ne7.ip.correctRaster(block.channel2, rasterPhase, fillFraction);
                    end
                    block.channel1 = ne7.ip.correctMotion(block.channel1, xy);
                    block.channel2 = ne7.ip.correctMotion(block.channel2, xy);
                    sz = size(block.channel1);
                    for iSlice = 1:length(sliceKeys)
                        t1 = reshape(block.channel1(:,:,iSlice,:), [], sz(4));
                        t1 = cellfun(@(ix) mean(t1(ix,:),1)', pixels{iSlice}, 'uni', false);
                        t2 = reshape(block.channel2(:,:,iSlice,:), [], sz(4));
                        t2 = cellfun(@(ix) mean(t2(ix,:),1)', pixels{iSlice}, 'uni', false);
                        traces{iSlice} = cat(1,traces{iSlice},cat(2,t2{:})./cat(2,t1{:}));
                    end
                else
                    block = getfield(reader.read(1, 1:reader.nSlices, blockSize),'channel1'); %#ok<GFLD>
                    xy = xymotion(:,:,1:size(block,4));
                    xymotion(:,:,1:size(block,4)) = [];
                    if strcmp(reader.hdr.scanMode, 'bidirectional')
                        block = ne7.ip.correctRaster(block, rasterPhase, fillFraction);
                    end
                    block = ne7.ip.correctMotion(block, xy);
                    sz = size(block);
                    for iSlice = 1:length(sliceKeys)
                        t = reshape(block(:,:,iSlice,:), [], sz(4));
                        t = cellfun(@(ix) mean(t(ix,:),1)', pixels{iSlice}, 'uni', false);
                        traces{iSlice} = cat(1,traces{iSlice},cat(2,t{:}));
                    end
                end
                
                fprintf('%5d frames\n', size(traces{1},1))
            end
            
            disp 'saving traces...'
            for iSlice = 1:nSlices
                tuple = sliceKeys(iSlice);
                for iTrace=1:size(traces{iSlice},2)
                    if ~isempty(pixels{iSlice}{iTrace})
                        tuple.trace_id = masks{iSlice}(pixels{iSlice}{iTrace}(1));
                        tuple.ca_trace = single(traces{iSlice}(:,iTrace));
                        self.insert(tuple)
                    end
                end
            end
        end
    end
end
