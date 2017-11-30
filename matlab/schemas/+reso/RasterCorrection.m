%{
# raster correction for bidirectional resonant scans
-> reso.ScanInfo
-> reso.CorrectionChannel
---
template                    : longblob                      # average frame from the middle of the movie
raster_phase                : float                         # difference between expected and recorded scan angle
%}


classdef RasterCorrection < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end