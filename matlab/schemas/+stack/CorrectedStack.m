%{
# all slices of each stack after corrections.
-> stack.StitchingVolume
---
x                           : float                         # (px) center of volume in a volume-wise coordinate system
y                           : float                         # (px) center of volume in a volume-wise coordinate system
z                           : float                         # (um) initial depth in the motor coordinate system
px_height                   : smallint                      # lines per frame
px_width                    : smallint                      # pixels per line
px_depth                    : smallint                      # number of slices
um_height                   : float                         # height in microns
um_width                    : float                         # width in microns
um_depth                    : float                         # depth in microns
%}


classdef CorrectedStack < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end