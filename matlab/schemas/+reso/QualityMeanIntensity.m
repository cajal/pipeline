%{
# mean intensity values across time
-> reso.Quality
-> `pipeline_shared`.`#slice`
-> `pipeline_shared`.`#channel`
---
intensities                 : longblob                      # 
%}


classdef QualityMeanIntensity < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end