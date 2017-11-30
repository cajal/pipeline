%{
# general data about mesoscope scans
-> experiment.Scan
-> meso.Version
---
nfields                     : tinyint                       # number of fields
nchannels                   : tinyint                       # number of channels
nframes                     : int                           # number of recorded frames
nframes_requested           : int                           # number of requested frames (from header)
x                           : float                         # (um) ScanImage's 0 point in the motor coordinate system
y                           : float                         # (um) ScanImage's 0 point in the motor coordinate system
fps                         : float                         # (Hz) frames per second
bidirectional               : tinyint                       # true = bidirectional scanning
usecs_per_line              : float                         # microseconds per scan line
fill_fraction               : float                         # raster scan temporal fill fraction (see scanimage)
nrois                       : tinyint                       # number of ROIs (see scanimage)
%}


classdef ScanInfo < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end