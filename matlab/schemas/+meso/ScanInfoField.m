%{
# field-specific scan information
-> meso.ScanInfo
-> shared.Field
---
px_height                   : smallint                      # height in pixels
px_width                    : smallint                      # width in pixels
um_height                   : float                         # height in microns
um_width                    : float                         # width in microns
x                           : float                         # (um) center of field in the motor coordinate system
y                           : float                         # (um) center of field in the motor coordinate system
z                           : float                         # (um) absolute depth with respect to the surface of the cortex
delay_image                 : longblob                      # (ms) delay between the start of the scan and pixels in this field
roi                         : tinyint                       # ROI to which this field belongs
%}


classdef ScanInfoField < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end