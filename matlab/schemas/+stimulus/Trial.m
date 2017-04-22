%{
# visual stimulus trial
-> experiment.Scan
trial_idx                   : int                           # trial index within sessions
---
-> stimulus.Condition
flip_times                  : longblob                      # (s) row array of flip times
last_flip                   : int unsigned                  # the last flip number in this trial
trial_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
%}


classdef Trial < dj.Manual
end
