%{
# 
-> meso.Fluorescence
-> meso.SegmentationMask
---
trace                       : longblob                      # 
%}


classdef FluorescenceTrace < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end