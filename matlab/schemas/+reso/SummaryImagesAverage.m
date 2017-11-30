%{
# l6-norm of each pixel across time
-> reso.SummaryImages
---
average_image               : longblob                      # 
%}


classdef SummaryImagesAverage < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end