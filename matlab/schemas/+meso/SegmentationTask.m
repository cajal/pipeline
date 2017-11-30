%{
# defines the target of segmentation and the channel to use
-> experiment.Scan
-> shared.Field
-> shared.Channel
-> shared.SegmentationMethod
---
-> experiment.Compartment
%}


classdef SegmentationTask < dj.Manual
end