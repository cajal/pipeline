%{
# inferred background components
-> reso.SegmentationCNMF
---
masks                       : longblob                      # array (im_height x im_width x num_background_components)
activity                    : longblob                      # array (num_background_components x timesteps)
%}


classdef SegmentationCNMFBackground < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end