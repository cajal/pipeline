%{
# union of ROIs from a stack (usually one per stack)
-> stack.Corrections
volume_id                   : tinyint                       # id of this volume
---
x                           : float                         # (um) center of ROI in a volume-wise coordinate system
y                           : float                         # (um) center of ROI in a volume-wise coordinate system
z                           : float                         # (um) initial depth in a volume-wise coordinate system
px_height                   : smallint                      # lines per frame
px_width                    : smallint                      # pixels per line
px_depth                    : smallint                      # number of slices
um_height                   : float                         # height in microns
um_width                    : float                         # width in microns
um_depth                    : float                         # depth in microns
%}


classdef CorrectionsStitched < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end