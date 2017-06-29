%{
# directional drift trials
-> tuning.Directional
drift_trial     : smallint               # trial index
---
-> vis.Trial
direction                   : float                         # (degrees) direction of drift
onset                       : double                        # (s) onset time in rf.Sync times
offset                      : double                        # (s) offset time in rf.Sync times
%}


classdef DirectionalTrial < dj.Part
    properties(SetAccess=protected)
        master = tuning.Directional
    end
end
