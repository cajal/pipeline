%{
# coordinates for each ROI in the stitched volume
-> stack.Corrections
-> stack.StackInfoROI
---
-> stack.CorrectionsStitched
x                           : float                         # (pixels) center of ROI in a volume-wise coordinate system
y                           : float                         # (pixels) center of ROI in a volume-wise coordinate system
z                           : float                         # (pixels) initial depth in a volume-wise coordinate system
%}


classdef CorrectionsROICoordinates < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end