%{
# Used for the webinterface
-> `pipeline_experiment`.`scan`
-> `pipeline_shared`.`#slice`
-> `pipeline_shared`.`#channel`
%}


classdef DoNotSegment < dj.Manual
end