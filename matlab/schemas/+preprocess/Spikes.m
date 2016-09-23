%{
preprocess.Spikes (computed) # infer spikes from calcium traces
-> preprocess.ComputeTraces
-> preprocess.SpikeMethod
---
%}


classdef Spikes < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.ComputeTraces*preprocess.SpikeMethod  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end