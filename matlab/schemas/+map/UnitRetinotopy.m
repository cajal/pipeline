%{
# brain area membership of cells
-> fuse.ScanSetUnit
-> anatomy.RefMap
-> map.RetMap
---
vret                        : double                        # vertical retinotopic angle
hret                        : double                        # horizontal retinotopic angle
%}


classdef UnitRetinotopy <  dj.Computed
    
    properties
        keySource = proj(anatomy.RefMap)*map.RetMap*fuse.ScanSetUnit&'unit_id=1'
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % get scan info
            key = rmfield(key, 'unit_id');
            setup = fetch1(experiment.Scan * experiment.Session & key,'rig');
            field_keys = fetch(fuse.ScanSet*map.RetMap*proj(anatomy.RefMap) & key);
            
            % process cells from each field
            for field_key = field_keys'
                
                % build image with area masks
                ret_key = [];
                ret_key.animal_id = field_key.animal_id;
                ret_key.ret_idx = field_key.ret_idx;
                Hret = fetch1(map.OptImageBar & (map.RetMapScan & ret_key) & 'axis = "horizontal"','ang');
                Vret = fetch1(map.OptImageBar & (map.RetMapScan & ret_key) & 'axis = "vertical"','ang');
                
                % find corresponding mask area
                Hret = filterMask(anatomy.FieldCoordinates & field_key, Hret);
                Vret = filterMask(anatomy.FieldCoordinates & field_key, Vret);
                
                % fetch cell coordinates
                if strcmp(setup,'2P4')
                    [px,wt,keys] = fetchn(meso.SegmentationMask*meso.ScanSetUnit & field_key,'pixels','weights');
                    
                    
                else
                    [px,wt,keys] = fetchn(reso.SegmentationMask*reso.ScanSetUnit & field_key,'pixels','weights');
                    keys = rmfield(keys,{'field','mask_id','channel'});
                    
                end
                [keys.ref_idx] = deal(key.ref_idx);
                [keys.ret_idx] = deal(key.ret_idx);
                
                % insert each cell
                for imask = 1:length(keys)
                    % get mask position
                    
                    tuple = keys(imask);
                    idx= px{imask}<=numel(Vret);
                    tuple.vret =  sum(Vret(px{imask}(idx)).*wt{imask}(idx))/sum(wt{imask}(idx));
                    tuple.hret = sum(Hret(px{imask}(idx)).*wt{imask}(idx))/sum(wt{imask}(idx));
                    
                    % insert
                    self.insert(tuple);
                end
            end
            
        end
    end
    
end