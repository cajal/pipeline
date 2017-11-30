%{
# average temporal correlation between each pixel and its eight neighbors
-> meso.SummaryImages
---
correlation_image           : longblob                      # 
%}


classdef SummaryImagesCorrelation < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end