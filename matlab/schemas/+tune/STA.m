%{
# Spike-triggered average receptive field maps
-> stimulus.Sync
-> fuse.Activity
-> tune.StimulusType
---
nbins                       : tinyint                       # number of bins
bin_size                    : decimal(3,3)                  # (s)
total_duration              : decimal(6,2)                  # total duration of included trials
vmax                        : float                         # correlation value of int8 level at 127
%}


classdef STA < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end