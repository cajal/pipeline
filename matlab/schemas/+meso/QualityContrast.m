%{
# difference between 99 and 1 percentile across time
-> meso.Quality
-> shared.Field
-> shared.Channel
---
contrasts                   : longblob                      # 
%}


classdef QualityContrast < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end