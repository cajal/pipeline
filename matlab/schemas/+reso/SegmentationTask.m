%{
# defines the target of segmentation and the channel to use
-> experiment.Scan
-> `pipeline_shared`.`#slice`
-> `pipeline_shared`.`#channel`
-> `pipeline_shared`.`#segmentation_method`
---
-> experiment.Compartment
%}


classdef SegmentationTask < dj.Manual
end