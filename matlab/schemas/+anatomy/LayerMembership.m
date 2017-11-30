%{
#
-> fuse.ScanSetUnit
---
-> anatomy.Layer
%}


classdef LayerMembership < dj.Computed
    
    properties
        keySource = experiment.Scan * shared.SegmentationMethod * shared.PipelineVersion & fuse.ScanDone & anatomy.FieldCoordinates
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            % get scan information
            field_keys = fetch(anatomy.FieldCoordinates & (fuse.ScanSetUnit & key));
            
            % get layer information
            [layers, zstart, zend] = fetchn(anatomy.Layer,'layer','z_start','z_end');
            
            % process all neurons from each field separately 
            for field_key = field_keys'
                % get depth for field
                depth = fetch1(anatomy.FieldCoordinates & (fuse.ScanSetUnit & field_key),'field_depth');
                
                % get all masks
                mask_keys = fetch(fuse.ScanSetUnit & field_key);
                
                % get correct layer
                idx = depth>zstart & depth<=zend;
                if any(idx)
                    layer = layers{depth>zstart & depth<=zend};
                else
                    layer = 'unset';
                end
                
                % insert
                [mask_keys.layer] = deal(layer);
                self.insert(mask_keys)
            end
        end
    end
end