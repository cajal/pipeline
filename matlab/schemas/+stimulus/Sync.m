%{
# synchronization of stimulus to scan
-> experiment.Scan
---
signal_start_time           : double                        # (s) signal start time on stimulus clock
signal_duration             : double                        # (s) signal duration on stimulus time
frame_times=null            : longblob                      # times of frames and slices on stimulus clock
sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
%}

classdef Sync < dj.Imported
    
    
    methods
        function source = makeKeySource(~)
            source = experiment.Scan & stimulus.Trial;
        end
        
    end
    
    methods(Static)
        function migrate
            % migrate from the legacy schema vis
            % This is incremental: can be called multiple times
            missing = preprocess.Sync - proj(stimulus.Sync);
            ignore_extra = dj.set('ignore_extra_insert_fields', true);
            insert(stimulus.Sync, missing.fetch('*'))
            dj.set('ignore_extra_insert_fields', ignore_extra);
        end
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            error 'not done yet'
            self.insert(key)
        end
    end
    
end