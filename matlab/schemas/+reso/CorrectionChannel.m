%{
# channel to use for raster and motion correction
-> `pipeline_experiment`.`scan`
-> `pipeline_shared`.`#slice`
---
-> `pipeline_shared`.`#channel`
%}


classdef CorrectionChannel < dj.Manual
end