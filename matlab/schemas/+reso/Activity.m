%{
# activity inferred from fluorescence traces
-> reso.ScanSet
-> shared.SpikeMethod
---
activity_time=CURRENT_TIMESTAMP: timestamp                  # automatic
%}


classdef Activity < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end