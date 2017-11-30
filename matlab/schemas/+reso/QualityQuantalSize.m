%{
# quantal size in images
-> reso.Quality
-> shared.Field
-> shared.Channel
---
min_intensity               : int                           # min value in movie
max_intensity               : int                           # max value in movie
quantal_size                : float                         # variance slope, corresponds to quantal size
zero_level                  : int                           # level corresponding to zero (computed from variance dependence)
quantal_frame               : longblob                      # average frame expressed in quanta
%}


classdef QualityQuantalSize < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end