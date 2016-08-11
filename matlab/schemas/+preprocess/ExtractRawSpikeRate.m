%{
preprocess.ExtractRawSpikeRate (imported) # spike trace extracted while segmentation
-> preprocess.ExtractRawTrace
---
spike_trace                 : longblob                      # 
%}


classdef ExtractRawSpikeRate < dj.Relvar
	methods

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			self.insert(key)
		end
	end

end