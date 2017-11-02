%{
# my newest table
-> preprocess.PrepareGalvoAverageFrame
---
frame_contrast              : float                         # compute frame contrast
%}


classdef FrameConstrast < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end