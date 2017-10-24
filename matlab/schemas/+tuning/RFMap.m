%{
# spatiotemporal receptive field map
-> tuning.RF
-> preprocess.SpikesRateTrace
---
scale                       : float                         # receptive field scale
map                         : longblob                      # int8 data map scaled by scale
%}


classdef RFMap < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end