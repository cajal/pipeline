%{
pre.SegmentMask (computed) # mask of a segmented cell
-> pre.Segment
mask_id                     : int # id of the mask
-----
mask_pixels                  : longblob # indices into the image in column major (Fortran) order
mask_weights                 : longblob # weights of the mask at the indices above
%}

classdef SegmentMask < dj.Relvar
    methods
        
        function makeTuples(self, key)
            switch fetch1(pre.SegmentMethod & key,'method_name')
                case 'manual'
                    mask = fetch1(pre.ManualSegment & key, 'mask');
                    regions = regionprops(bwlabel(mask, 4),'PixelIdxList'); %#ok<MRPBW>
                    regions =  dj.struct.rename(regions, 'PixelIdxList', 'mask_pixels');
                    tuples = arrayfun(@(i) setfield(regions(i), 'mask_id', i), 1:length(regions)); %#ok<SFLD>
                    tuples = dj.struct.join(key, tuples');
                    [tuples.mask_weights] = deal(1);
                    
                    self.insert(tuples)
                case 'nmf'
                    error 'Not implemented yet'
                otherwise
                    error 'Unknown segmentation method'
            end
            
            
        end
    end
    
end