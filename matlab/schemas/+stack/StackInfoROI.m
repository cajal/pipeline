%{
# 3-D volumes that compose this stack (usually tiled to form a bigger fov)
-> stack.StackInfo
roi_id                      : tinyint                       # same as ScanImage's
---
-> experiment.StackFilename
field_ids                   : blob                          # list of field_ids (0-index) sorted from shallower to deeper
x                           : float                         # (um) center of ROI in the motor coordinate system
y                           : float                         # (um) center of ROI in the motor coordinate system
z                           : float                         # (um) initial depth in the motor coordinate system
px_height                   : smallint                      # lines per frame
px_width                    : smallint                      # pixels per line
px_depth                    : smallint                      # number of slices
um_height                   : float                         # height in microns
um_width                    : float                         # width in microns
um_depth                    : float                         # depth in microns
nframes                     : smallint                      # number of recorded frames per plane
fps                         : float                         # (Hz) volumes per second
bidirectional               : tinyint                       # true = bidirectional scanning
%}


classdef StackInfoROI < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end