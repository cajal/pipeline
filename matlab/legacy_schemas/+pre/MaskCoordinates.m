%{
pre.MaskCoordinates (computed) # mask center of mass of a segmented cell
-> pre.SegmentMask
-----
xloc                 : double # x location in micro meters relative to the frame start
yloc                 : double # y location in micro meters relative to the frame start
zloc                 : double # z location in micro meters relative to the surface
%}

classdef MaskCoordinates < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.Segment
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            % get scan info
            [masks,weights,mask_id] = fetchn(pre.SegmentMask & key,'mask_pixels','mask_weights','mask_id');
            reader = pre.getReader(key);
            slice_pos = reader.header.hFastZ_userZs;
            [px_width,px_height,width,height] = fetch1(pre.ScanInfo & key,...
                'px_width','px_height','um_width','um_height');
            depth = fetch1(rf.Scan & key, 'depth');
            
            for imask = 1:length(mask_id);
                
                % get mask position
                mask = masks{imask};
                im = zeros(px_width,px_height);
                im(mask) = weights{imask};
                labeledImage = logical(true(size(im)));
                measurements = regionprops(labeledImage, im, 'WeightedCentroid');
                px_centerOfMass = measurements.WeightedCentroid;
                centerOfMass = px_centerOfMass./size(im).*[width,height] - [width,height]/2;
                
                % get slice possition (um)
                slice_position = depth - slice_pos(key.slice);
                
                % insert
                key.mask_id = mask_id(imask);
                key.xloc = centerOfMass(1);
                key.yloc = centerOfMass(2);
                key.zloc = slice_position;
                self.insert(key);
            end
        end
    end
end




