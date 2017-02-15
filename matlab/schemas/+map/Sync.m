%{
map.Sync (imported) # 
-> experiment.Scan
---
-> vis.Session
first_trial                 : int                           # first trial index from psy.Trial overlapping recording
last_trial                  : int                           # last trial index from psy.Trial overlapping recording
signal_start_time           : double                        # (s) signal start time on stimulus clock
signal_duration             : double                        # (s) signal duration on stimulus time
frame_times=null            : longblob                      # times of frames and slices
sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
%}


classdef Sync < dj.Relvar 
    
    methods
        
        function makeTuples(self, tuple)
            self.insert(tuple)
        end
    end
    
end
