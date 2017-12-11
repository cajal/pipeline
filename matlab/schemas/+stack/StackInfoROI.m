%{
# 3-D volumes that compose this stack (usually tiled to form a bigger fov)
-> stack.StackInfo
roi_id                      : tinyint                       # same as ScanImage's
---
-> experiment.StackFilename
field_ids                   : blob                          # list of field_ids (0-index) sorted from shallower to deeper
roi_x                       : float                         # (um) center of ROI in the motor coordinate system
roi_y                       : float                         # (um) center of ROI in the motor coordinate system
roi_z                       : float                         # (um) initial depth in the motor coordinate system
roi_px_height               : smallint                      # lines per frame
roi_px_width                : smallint                      # pixels per line
roi_px_depth                : smallint                      # number of slices
roi_um_height               : float                         # height in microns
roi_um_width                : float                         # width in microns
roi_um_depth                : float                         # depth in microns
nframes                     : smallint                      # number of recorded frames per plane
fps                         : float                         # (Hz) volumes per second
bidirectional               : tinyint                       # true = bidirectional scanning
is_slow                     : tinyint                       # whether all frames in one depth were recorded before moving to the next
%}


classdef StackInfoROI < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end