%{
vis.Trial (manual) # visual stimulus trial
-> vis.Session
trial_idx       : int                    # trial index within sessions
---
-> vis.Condition
flip_times                  : mediumblob                    # (s) row array of flip times
last_flip_count             : int unsigned                  # the last flip number in this trial
trial_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
%}


classdef Trial < dj.Relvar
end