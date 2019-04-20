%{
# visual stimulus trial buffer control
twop_setup                  : varchar(256)                  # two photon setup name, e.g. 2P1
setup                       : varchar(256)                  # Setup name of behavior/stim computer, e.g. at-stim01
---
cmd                         : smallint                      # =0 if need sync, =1 if need sync&cleaning
state                       : smallint                      # =0 if sync in progress, ==1 if complete sync&clean
trial_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
%}


classdef TrialBufferControl < dj.Manual
end
