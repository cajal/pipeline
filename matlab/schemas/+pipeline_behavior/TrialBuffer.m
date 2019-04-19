%{
# visual stimulus trial buffering
animal_id                   : int
session                     : smallint
scan_idx                    : smallint
trial_idx                   : int                           # trial index within sessions
---
condition_hash              : char(20)
flip_times                  : longblob                      # (s) row array of flip times
last_flip                   : int unsigned                  # the last flip number in this trial
trial_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
%}


classdef TrialBuffer < dj.Manual
end
