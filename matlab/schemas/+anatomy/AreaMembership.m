%{
# brain area membership of cells
-> fuse.ScanSetUnit
---
-> experiment.BrainArea
%}


classdef AreaMembership <  dj.Imported
    
    properties
        keySource = fuse.ScanSetUnit & map.AreaMask
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % get scan info
            [masks,weights,keys] = fetchn( ...
                preprocess.ExtractRawGalvoROI & (preprocess.Slice  * preprocess.ExtractRaw & key),...
                'mask_pixels','mask_weights');
            [px_width,px_height] = fetch1(preprocess.PrepareGalvo & key,...
                'px_width','px_height');
            [area_masks, areas] = fetchn(map.AreaMask & key,'mask','area');
            area_mask = zeros(size(area_masks{1}));
            for iarea=1:length(area_masks)
                area_mask(area_masks{iarea}>0) = iarea;
            end
            
            for imask = 1:length(keys)
                
                % get mask position
                mask = masks{imask};
                im = zeros(px_width,px_height);
                im(mask) = weights{imask};
                labeledImage = true(size(im));
                measurements = regionprops(labeledImage, im, 'WeightedCentroid');
                px_centerOfMass = measurements.WeightedCentroid;
                area_idx = area_mask(round(px_centerOfMass(1)),round(px_centerOfMass(2)));

                tuple = keys(imask);
                if area_idx>0
                    tuple.brain_area = areas{area_idx};
                else
                    tuple.brain_area = 'unknown';
                end
                self.insert(tuple);
            end
            
        end
    end
    
end