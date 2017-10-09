%{
# 16-part summary of the scan (mean of 16 blocks)
-> meso.Quality
-> `pipeline_shared`.`#field`
-> `pipeline_shared`.`#channel`
---
summary                     : longblob                      # h x w x 16
%}


classdef QualitySummaryFrames < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end