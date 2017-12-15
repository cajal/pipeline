%{
# coordinates for each ROI in the stitched volume
-> stack.Stitching
-> stack.MotionCorrection
---
-> stack.StitchingVolume
stitch_xs                   : blob                          # (px) center of each slice in the volume-wise coordinate system
stitch_ys                   : blob                          # (px) center of each slice in the volume-wise coordinate system
stitch_z                    : float                         # (um) initial depth in the motor coordinate system
%}


classdef StitchingROICoordinates < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end