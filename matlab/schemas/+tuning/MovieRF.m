%{
# spike-triggered average of receptive fields
-> preprocess.Sync
-> preprocess.Spikes
---
nbins                       : smallint                      # temporal bins
bin_size                    : float                         # (ms) temporal bin size
degrees_x                   : float                         # degrees along x
degrees_y                   : float                         # degrees along y
stim_duration               : float                         # (s) total stimulus duration
%}


classdef MovieRF < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end