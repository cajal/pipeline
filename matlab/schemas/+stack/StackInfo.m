%{
# master table with general data about the stacks
-> experiment.Stack
-> stack.Version
---
nrois                       : tinyint                       # number of ROIs
nchannels                   : tinyint                       # number of channels
z_step                      : float                         # (um) distance in z between adjacent slices (always positive)
fill_fraction               : float                         # raster scan temporal fill fraction (see scanimage)
%}


classdef StackInfo < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end