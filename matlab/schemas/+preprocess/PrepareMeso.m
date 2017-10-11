%{
# basic information about resonant microscope scans, raster correction
-> preprocess.Prepare
---
nframes_requested           : int                           # number of volumes (from header)
nframes                     : int                           # frames recorded
px_width                    : smallint                      # pixels per line
px_height                   : smallint                      # lines per frame
um_width                    : float                         # width in microns
um_height                   : float                         # height in microns
bidirectional               : tinyint                       # 1=bidirectional scanning
fps                         : float                         # (Hz) frames per second
zoom                        : decimal(4,1)                  # zoom factor
dwell_time                  : float                         # (us) microseconds per pixel per frame
nchannels                   : tinyint                       # number of recorded channels
nslices                     : tinyint                       # number of slices
nrois                       : tinyint                       # numner of ROIs
slice_pitch                 : float                         # (um) distance between slices
fill_fraction               : float                         # raster scan temporal fill fraction (see scanimage)
preview_frame               : longblob                      # raw average frame from channel 1 from an early fragment of the movie
raster_phase                : float                         # shift of odd vs even raster lines
%}


classdef PrepareMeso < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end