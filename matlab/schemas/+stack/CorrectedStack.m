%{
# all slices of each stack after corrections.
<<<<<<< HEAD
-> stack.CorrectionsStitched
=======
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
>>>>>>> 0bd758669df75014d62df8c0d09157b457ef65ed
%}


classdef CorrectedStack < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
<<<<<<< HEAD
			 self.insert(key)
=======
% 			 self.insert(key)
>>>>>>> 0bd758669df75014d62df8c0d09157b457ef65ed
		end
	end

end