%{
# brain area membership of cells
-> fuse.ScanSetUnit
---
-> anatomy.Area
%}


classdef AreaMembership <  dj.Computed
    
    properties
        keySource = proj(experiment.Scan) *  proj(shared.SegmentationMethod) *  proj(shared.PipelineVersion) & fuse.ScanDone & anatomy.AreaMask
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % get scan info
            setup = fetch1(experiment.Scan * experiment.Session & key,'rig');
            field_keys = fetch(fuse.ScanSet & anatomy.AreaMask & key);
            
            % process cells from each field
            for field_key = field_keys'
                
                % build image with area masks
                [area_masks, areas] = fetchn(anatomy.AreaMask & field_key,'mask','brain_area');
                area_mask = zeros(size(area_masks{1}));
                for iarea=1:length(area_masks)
                    area_mask(area_masks{iarea}>0) = iarea;
                end
             
                % fetch cell coordinates
                if strcmp(setup,'2P4')
                    [px_width, px_height, keys] = ...
                        fetchn((meso.ScanSetUnitInfo & (fuse.ScanSetUnit & field_key)) * ...
                        proj(meso.ScanInfoField & field_key,'px_height','px_width'),...
                        'px_x','px_y');
                    keys = rmfield(keys,'field');
                else
                    [px_width, px_height, keys] = ...
                        fetchn((reso.ScanSetUnitInfo & (fuse.ScanSetUnit & field_key)) * ...
                        proj(reso.ScanInfo & field_key,'px_height','px_width'),...
                        'px_x','px_y');
                end

                % insert each cell
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
    
end