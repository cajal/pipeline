%{
# channel to use for raster and motion correction
-> `pipeline_experiment`.`stack`
---
-> `pipeline_shared`.`#channel`
%}


classdef CorrectionChannel < dj.Manual
end