%{
preprocess.Spikes (computed) # infer spikes from calcium traces
-> preprocess.ComputeTraces
-> preprocess.SpikeMethod
---
%}


classdef Spikes < dj.Relvar & dj.AutoPopulate

	properties
        % NMF spikes (method 5) can be copied via MATLAB, STM spike detection is performed in Python
		popRel = preprocess.ComputeTraces*preprocess.SpikeMethod  & preprocess.ExtractRawSpikeRate & 'spike_method=5';
	end

	methods(Access=protected)

		function makeTuples(self, key)
            
            % Copy NMF spikes from preprocess.ExtractRawSpikeRate
            tuples = fetch(preprocess.ExtractRawSpikeRate & key,'spike_trace->rate_trace','*');
            tuples = rmfield(tuples,'channel');
            [tuples.spike_method] = deal(5);
            
			self.insert(key)
            insert(preprocess.SpikesRateTrace,tuples);
            
            
		end
	end

end