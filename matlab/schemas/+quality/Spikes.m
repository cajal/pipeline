%{
quality.Spikes (computed) # 
-> preprocess.Spikes
---
leading_nans                : tinyint                       # whether or not any of the traces has leading nans
trailing_nans               : tinyint                       # whether or not any of the traces has trailing nans
stimulus_nans               : tinyint                       # whether or not any of the traces has nans during the stimulus
nan_idx=null                : longblob                      # boolean array indicating where the nans are
stimulus_start              : int                           # start of the stimulus in matlab 1 based indices
stimulus_end                : int                           # end of the stimulus in matlab 1 based indices
%}


classdef Spikes < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.Spikes  
	end

	methods(Access=protected)

		function makeTuples(self, key)
            error 'populated in python'
		end
	end

end