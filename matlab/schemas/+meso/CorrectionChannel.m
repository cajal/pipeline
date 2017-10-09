%{
# channel to use for raster and motion correction
-> experiment.Scan
-> `pipeline_shared`.`#field`
---
-> `pipeline_shared`.`#channel`
%}


classdef CorrectionChannel < dj.Manual
end