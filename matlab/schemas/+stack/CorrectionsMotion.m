%{
# motion correction for each slice in the stack (frame-to-frame and slice-to-slice)
-> stack.Corrections
-> stack.StackInfoROI
---
y_shifts                    : longblob                      # y motion correction shifts (num_slices x num_frames)
x_shifts                    : longblob                      # x motion correction shifts (num_slices x num_frames)
y_aligns                    : longblob                      # isolated slice-to-slice alignment shifts (num_slices)
x_aligns                    : longblob                      # isolated slice-to-slice alignment shifts (num_slices)
%}


classdef CorrectionsMotion < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end