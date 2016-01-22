%{
pre.MaskAverageTrace (computed) # compute traces with masks
->pre.SelectedMask
-----
trace              : longblob # avg(mask.*scan)
%}

classdef MaskAverageTrace < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.NMFSegment() & pre.SelectedMask()
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            [masks, keys] = pre.SelectedMask().load_masks(key);
            reader = pre.getReader(key, '/tmp');
            assert(reader.nslices == 1, 'schema only supports one slice at the moment');
            
            maxT = reader.nframes;
            blockSize = min(maxT, 10000);
            
            
            pointer = 1;
            traces = zeros(size(masks,3), maxT);
            while pointer < maxT
                step =  min(blockSize, maxT-pointer+1);
                frames = pointer:pointer+step-1;
            
                fprintf('Reading frames  %i:%i of maximally %i \n', ...
                        frames(1), frames(end), maxT);
                block = pre.load_corrected_block(key, reader, frames);
                for i = 1:size(masks,3)
                    traces(i, frames) = squeeze(sum(sum(bsxfun(@times, block, masks(:,:,i)), 1),2));
                end
                pointer = pointer + step;
            end
            
            for i = 1:length(keys)
                mykey = keys(i);
                mykey.trace = traces(i,:);
                self.insert(mykey);
            end
            
        end
        
    end
    
end
