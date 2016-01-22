%{
rf.Segment (imported) # 2d cell segmentation
-> rf.Align
-> rf.VolumeSlice
---
mask                        : longblob                      # image with roi labels
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}

classdef Segment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = rf.Align & rf.ManualSegment
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            if fetch1(rf.Scan & key,'site')==0                % the site is not assigned, just copy the mask from manual segmentation
                tuples = fetch(rf.ManualSegment & key,'*');
                for i=1:length(tuples)
                    tuples(i).mask = bwlabel(tuples(i).mask,4);
                end
            else
                % if the site is assigned, then take map from rf.SiteSegment
                assert(count(rf.SiteSegment & key)>0, 'Please populate rf.SiteSegment first')
                tuples = fetch(rf.SiteSegment & key, 'label_mask->mask');
            end
            self.insert(tuples)
            makeTuples(rf.Trace, key)
        end
    end
end
