%{
stimulus.SelectCondition (manual) # conditions that are generated but not yet shown -- helpful for online analysis but unnecessary once stimuli have been shown.
-> experiment.Scan
-> stimulus.Condition
%}

classdef SelectCondition < dj.Relvar
end