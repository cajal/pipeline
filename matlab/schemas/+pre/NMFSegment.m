%{
pre.NMFSegment (imported) # 2d cell segmentations computed with NMF
-> pre.AlignMotion
-> rf.VolumeSlice
-> pre.Settings
---
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}



classdef NMFSegment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.AlignMotion * pre.Settings
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            [d1, d2, nslices] = self.get_resolution(key);
            assert(nslices==1, 'This schema only supports one slice.')
            
            key.slice_num = 1;
            insert(self, key);
            patch_size=128;
            for i = 1:patch_size/2:d1-patch_size+1
                for j = 1:patch_size/2:d2-patch_size+1
                    key.rstart = i;
                    key.rend = i + patch_size - 1;
                    key.cstart = j;
                    key.cend = j + patch_size - 1;
                    pre.Tesselation().insert(key);
                end
            end
        end
    end
    
    methods(Static)
        %%------------------------------------------------------------
        function [d1, d2, nslices] = get_resolution(key)
            [d1,d2, nslices] = fetch1(pre.ScanInfo & key, 'px_height', 'px_width','nslices');
        end
        %%------------------------------------------------------------
    end
    
end
