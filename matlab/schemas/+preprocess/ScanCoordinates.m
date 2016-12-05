%{
preprocess.ScanCoordinates (imported) # mask center of mass of a segmented cell
-> preprocess.ExtractRawGalvoSegmentation
-----
xpos                 : double # x motor location in micro meters 
ypos                 : double # y motor location in micro meters 
zpos                 : double # z motor location in micro meters 
%}

classdef ScanCoordinates < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = preprocess.ExtractRawGalvoSegmentation
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            % get scan info
            [masks,weights,mask_id] = fetchn(preprocess.ExtractRawGalvoROI & key,'mask_pixels','mask_weights','trace_id');
            reader = preprocess.getGalvoReader(key);
            slice_pos = reader.header.hFastZ_userZs;
            scan_pos = reader.header.hMotors_motorPosition;
            [px_width,px_height,width,height] = fetch1(preprocess.PrepareGalvo & key,...
                'px_width','px_height','um_width','um_height');
            depth = fetch1(experiment.Scan & key, 'depth');
            
            tuple = key;
            tuple.xpos = scan_pos(1);
            tuple.ypos = scan_pos(2);
            tuple.zpos = scan_pos(3);
            
            self.insert(tuple);
            
            for imask = 1:length(mask_id)
                
                % get mask position
                mask = masks{imask};
                im = zeros(px_width,px_height);
                im(mask) = weights{imask};
                labeledImage = true(size(im));
                measurements = regionprops(labeledImage, im, 'WeightedCentroid');
                px_centerOfMass = measurements.WeightedCentroid;
                centerOfMass = px_centerOfMass./size(im).*[width,height] - [width,height]/2;
                
                % get slice possition (um)
                slice_position = depth + slice_pos(key.slice);
                
                % insert
                key.trace_id = mask_id(imask);
                key.xloc = centerOfMass(1);
                key.yloc = centerOfMass(2);
                key.zloc = slice_position;
                makeTuples(preprocess.MaskCoordinates,key);
            end
        end
    end
end