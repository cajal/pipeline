%{
# treadmill velocity synchronized to behavior clock
-> `pipeline_experiment`.`scan`
-> `pipeline_experiment`.`scan`
---
treadmill_raw               : longblob                      # raw treadmill counts
treadmill_time              : longblob                      # (secs) velocity timestamps in behavior clock
treadmill_vel               : longblob                      # (cm/sec) wheel velocity
treadmill_ts=CURRENT_TIMESTAMP: timestamp                   # 
%}


classdef Treadmill < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
		%	 self.insert(key)
		end
	end

end