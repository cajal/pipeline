%{
# threshold for masks created manually
-> reso.Segmentation
---
threshold_type                  : tinyint       # 1=in SD relative to mean, 2=absolute pixel value
threshold_value                 : float         # depending on type, this is the tresholding value
entry_time=CURRENT_TIMESTAMP    : timestamp     # time tuple was created/update
%}


classdef SegmentationMaskThreshold < dj.Manual
    methods(Access=protected)
        function makeTuples(self, key)
        end
    end
end