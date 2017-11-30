%{
# motion correction for galvo scans
-> meso.RasterCorrection
---
template                    : longblob                      # image used as alignment template
y_shifts                    : longblob                      # (pixels) y motion correction shifts
x_shifts                    : longblob                      # (pixels) x motion correction shifts
y_std                       : float                         # (um) standard deviation of y shifts
x_std                       : float                         # (um) standard deviation of x shifts
y_outlier_frames            : longblob                      # mask with true for frames with high y shifts (already corrected)
x_outlier_frames            : longblob                      # mask with true for frames with high x shifts (already corrected)
align_time=CURRENT_TIMESTAMP: timestamp                     # automatic
%}


classdef MotionCorrection < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end