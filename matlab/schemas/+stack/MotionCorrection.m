%{
# motion correction for each slice in the stack
-> stack.RasterCorrection
---
y_shifts                    : longblob                      # y motion correction shifts (num_slices x num_frames)
x_shifts                    : longblob                      # x motion correction shifts (num_slices x num_frames)
%}


classdef MotionCorrection < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end