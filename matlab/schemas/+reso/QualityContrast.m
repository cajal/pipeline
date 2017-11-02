%{
# difference between 99 and 1 percentile across time
-> reso.Quality
-> `pipeline_shared`.`#slice`
-> `pipeline_shared`.`#channel`
---
contrasts                   : longblob                      # 
%}


classdef QualityContrast < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end