%{
# unit type, coordinates and delay time
-> meso.ScanSetUnit
---
um_x                        : smallint                      # x-coordinate of centroid in motor coordinate system
um_y                        : smallint                      # y-coordinate of centroid in motor coordinate system
um_z                        : smallint                      # z-coordinate of mask relative to surface of the cortex
px_x                        : smallint                      # x-coordinate of centroid in the frame
px_y                        : smallint                      # y-coordinate of centroid in the frame
ms_delay                    : smallint                      # (ms) delay from start of frame to recording of this unit
%}


classdef ScanSetUnitInfo < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end