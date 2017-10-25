%{
# coordinates for each ROI in a volume
-> stack.StackInfoROI
---
-> stack.StitchingVolume
x                           : float                         # (pixels) center of ROI in a volume-wise coordinate system
y                           : float                         # (pixels) center of ROI in a volume-wise coordinate system
z                           : float                         # (pixels) initial depth in a volume-wise coordinate system
%}


classdef StitchingROICoordinates < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end