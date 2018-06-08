%{
# masks created manually
-> reso.Segmentation
---
%}


classdef SegmentationManual < dj.Computed
    
    properties
        popRel  = reso.SummaryImagesAverage
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            images = fetch1(reso.SummaryImagesAverage & key, 'average_image');
            % remove baseline
            images = images - min(images(:));
            masks = ne7.ui.paintMasks(images);
            assert(~isempty(masks), 'user aborted segmentation')
            key.segmentation_method = 1;
            r_key = rmfield(key,'pipe_version');
            r_key.compartment = 'unknown';
            insert(reso.SegmentationTask,r_key);
            insert(reso.Segmentation,key);
            self.insert(key)
            
            % Insert Masks
            unique_masks = unique(masks);
            key.mask_id = 0;
            for mask = unique_masks(unique_masks>0)'
                key.mask_id = key.mask_id+1;
                key.pixels = find(masks==mask);
                key.weights = ones(size(key.pixels));
                insert(reso.SegmentationMask,key)
            end
        end
        
    end
    
    methods
        function tranferMasks(self, target_key)
            r_key = fetch(reso.SegmentationTask & self);
            assert(length(unique([r_key.scan_idx]))==1,'Too many source keys!');
            assert(r_key(1).animal_id==target_key.animal_id & r_key(1).session==target_key.session,'Mask tranfer only supported within the same session');
            [r_key.scan_idx] = deal(target_key.scan_idx);
            disp 'Inserting segmentation task'
            insert(reso.SegmentationTask,r_key);
            key = fetch(reso.Segmentation & self);
            [key.scan_idx] = deal(target_key.scan_idx);
            disp 'Inserting segmentation'
            insert(reso.Segmentation,key);
            self.insert(key)
            mask_keys = fetch(reso.SegmentationMask & self,'*');
            [mask_keys.scan_idx] = deal(target_key.scan_idx);
            disp 'Inserting masks'
            insert(reso.SegmentationMask,mask_keys)
            disp 'done'
        end
    end
end

