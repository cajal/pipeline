%{
# mask produced by segmentation.
-> reso.Segmentation
mask_id                     : smallint                      # 
---
pixels                      : longblob                      # indices into the image in column major (Fortran) order
weights=null                : longblob                      # weights of the mask at the indices above
%}


classdef SegmentationMask < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end