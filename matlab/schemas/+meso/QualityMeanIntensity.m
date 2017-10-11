%{
# mean intensity values across time
-> meso.Quality
-> `pipeline_shared`.`#field`
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