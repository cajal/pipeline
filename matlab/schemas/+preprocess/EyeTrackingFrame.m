%{
# 
-> preprocess.EyeTracking
frame_id                    : int                           # frame id with matlab based 1 indexing
---
rotated_rect=null           : tinyblob                      # rotated rect (center, sidelength, angle) containing the ellipse
contour=null                : longblob                      # eye contour relative to ROI
center=null                 : tinyblob                      # center of the ellipse in (x, y) of image
major_r=null                : float                         # major radius of the ellipse
frame_intensity=null        : float                         # std of the frame
%}


classdef EyeTrackingFrame < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end