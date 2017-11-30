%{
# mean intensity values across time
-> reso.Quality
-> shared.Field
-> shared.Channel
---
intensities                 : longblob                      # 
%}


classdef QualityMeanIntensity < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end