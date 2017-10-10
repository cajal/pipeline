%{
# defines the target of segmentation and the channel to use
-> `pipeline_experiment`.`scan`
-> `pipeline_shared`.`#slice`
-> `pipeline_shared`.`#channel`
-> `pipeline_shared`.`#segmentation_method`
---
-> `pipeline_experiment`.`#compartment`
%}


classdef SegmentationTask < dj.Manual
end