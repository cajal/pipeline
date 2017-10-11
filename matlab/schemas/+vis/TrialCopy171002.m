%{
# visual stimulus trial
animal_id                   : int                           # id (internal to database)
psy_id                      : smallint unsigned             # unique psy session number
trial_idx                   : int                           # trial index within sessions
---
cond_idx                    : smallint unsigned             # condition index
flip_times                  : mediumblob                    # (s) row array of flip times
last_flip_count             : int unsigned                  # the last flip number in this trial
trial_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
%}


classdef TrialCopy171002 < dj.Manual
end