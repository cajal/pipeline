%{
pre.Segment (imported) # cell segmentation
-> pre.AlignMotion
-> pre.SegmentMethod
---
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}

classdef Segment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pro(pre.AlignMotion*pre.SegmentMethod) ...
            & (pre.ManualSegment*pre.SegmentMethod & 'method_name = "manual"' | ...
               pre.NMFSettings*pre.SegmentMethod & 'method_name = "nmf"');
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            self.insert(key)
            makeTuples(pre.SegmentMask, key)
        end
    end
end
