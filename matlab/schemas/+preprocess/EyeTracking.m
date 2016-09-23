%{
preprocess.EyeTracking (computed) # 
-> preprocess.Eye
-> preprocess.TrackingParameters
---
tracking_ts=CURRENT_TIMESTAMP: timestamp                    # automatic
%}


classdef EyeTracking < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.Eye*preprocess.TrackingParameters  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end