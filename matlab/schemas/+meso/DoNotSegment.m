%{
# field/channels that should not be segmented (used for web interface only)
-> experiment.Scan
-> `pipeline_shared`.`#field`
-> `pipeline_shared`.`#channel`
%}


classdef DoNotSegment < dj.Manual
end