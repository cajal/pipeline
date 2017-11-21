%{
# brain area membership of cells
-> fuse.ScanSetUnit
---
-> experiment.BrainArea
%}


classdef AreaMembership <  dj.Computed
    
    properties
        keySource = fuse.ScanSetUnit & anatomy.AreaMask
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % get scan info
            [area_masks, areas] = fetchn(anatomy.AreaMask & key,'mask','area');
            area_mask = zeros(size(area_masks{1}));
            for iarea=1:length(area_masks)
                area_mask(area_masks{iarea}>0) = iarea;
            end
            setup = fetch1(experiment.Scan * experiment.Session & key,'rig');
            
            if strcmp(setup,'2P4')
                [px_width, px_height, keys] = ...
                    fetchn((meso.ScanSetUnitInfo & (fuse.ScanSetUnit & key)) * ...
                    proj(meso.ScanInfoField & fuse.ScanSet &  (fuse.ScanSetUnit & key),'px_height','px_width'),...
                    'px_x','px_y');
            else
                [px_width, px_height, keys] = ...
                    fetchn((reso.ScanSetUnitInfo & (fuse.ScanSetUnit & key)) * ...
                    proj(reso.ScanInfo & fuse.ScanSet &  (fuse.ScanSetUnit & key),'px_height','px_width'),...
                    'px_x','px_y');
            end
            keys = rmfield(keys,'field');
          
            for imask = 1:length(keys)
                % get mask position
                area_idx = area_mask(round(px_height(imask)),round(px_width(imask)));
                tuple = keys(imask);
                if area_idx>0
                    tuple.brain_area = areas{area_idx};
                else
                    tuple.brain_area = 'unknown';
                end
                
                % insert
                self.insert(tuple);
            end
            
        end
    end
    
end