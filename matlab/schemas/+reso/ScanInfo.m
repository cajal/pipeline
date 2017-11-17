%{
# master table with general data about the scans
-> experiment.Scan
-> reso.Version
---
nfields                     : tinyint                       # number of slices
nchannels                   : tinyint                       # number of recorded channels
nframes                     : int                           # number of recorded frames
nframes_requested           : int                           # number of frames (from header)
px_height                   : smallint                      # lines per frame
px_width                    : smallint                      # pixels per line
um_height                   : float                         # height in microns
um_width                    : float                         # width in microns
x                           : float                         # (um) center of scan in the motor coordinate system
y                           : float                         # (um) center of scan in the motor coordinate system
fps                         : float                         # (Hz) frames per second
zoom                        : decimal(5,2)                  # zoom factor
bidirectional               : tinyint                       # true = bidirectional scanning
usecs_per_line              : float                         # microseconds per scan line
fill_fraction               : float                         # raster scan temporal fill fraction (see scanimage)
%}


classdef ScanInfo < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end