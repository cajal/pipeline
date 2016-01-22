%{
pre.Segment (imported) # cell segmentation
-> pre.AlignMotion
---
mask                        : longblob                      # image with roi labels
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}

classdef Segment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.AlignMotion & pre.ManualSegment
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            tuple = fetch(pre.ManualSegment & key, '*');
            assert(length(tuple)==1, 'we''ll deal with slices later')
            tuple.mask = bwlabel(tuple.mask, 4);
            self.insert(tuple)
            makeTuples(pre.Trace, key)
        end
    end
end
