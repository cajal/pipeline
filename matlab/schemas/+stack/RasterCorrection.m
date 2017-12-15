%{
# raster correction for bidirectional resonant scans
-> stack.CorrectionChannel
-> stack.StackInfoROI
---
raster_phase                : float                         # difference between expected and recorded scan angle
raster_std                  : float                         # standard deviation among raster phases in different slices
%}


classdef RasterCorrection < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end