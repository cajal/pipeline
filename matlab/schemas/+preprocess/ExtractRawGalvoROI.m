%{
preprocess.ExtractRawGalvoROI (imported) # region of interest produced by segmentation
-> preprocess.ExtractRawGalvoSegmentation
-> preprocess.ExtractRawTrace
---
mask_pixels                 : longblob                      # indices into the image in column major (Fortran) order
mask_weights=null           : longblob                      # weights of the mask at the indices above
%}


classdef ExtractRawGalvoROI < dj.Relvar
	methods

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			self.insert(key)
		end
	end

end