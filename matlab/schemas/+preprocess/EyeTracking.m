%{
# 
-> preprocess.Eye
-> preprocess.TrackingParameters
---
tracking_ts=CURRENT_TIMESTAMP: timestamp                    # automatic
%}


classdef EyeTracking < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end