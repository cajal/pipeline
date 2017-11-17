%{
# channel to use for raster and motion correction
-> experiment.Scan
-> shared.Field
---
-> shared.Channel
%}


classdef CorrectionChannel < dj.Manual
end