%{
preprocess.ExtractRawGalvoSegmentation (imported) # segmentation of galvo movies
-> preprocess.ExtractRaw
-> preprocess.Slice
---
segmentation_mask=null      : longblob                      # 
%}


classdef ExtractRawGalvoSegmentation < dj.Relvar
	methods

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			self.insert(key)
		end
	end

end