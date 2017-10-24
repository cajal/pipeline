%{
# Used for the webinterface
-> experiment.Scan
-> `pipeline_shared`.`#slice`
-> `pipeline_shared`.`#channel`
%}


classdef DoNotSegment < dj.Manual
end