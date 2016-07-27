%{
preprocess.SpikesRateTrace (computed) # Inferred
-> preprocess.Spikes
-> preprocess.ComputeTracesTrace
---
rate_trace=null             : longblob                      # leave null same as ExtractRaw.Trace
%}


classdef SpikesRateTrace < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.Spikes*preprocess.ComputeTracesTrace  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end