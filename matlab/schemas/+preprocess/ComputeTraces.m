%{
preprocess.ComputeTraces (computed) # compute traces
-> preprocess.ExtractRaw
---
%}


classdef ComputeTraces < dj.Relvar & dj.AutoPopulate

	properties
        % Twitch traces are populated in Python
		popRel = preprocess.ExtractRaw & (experiment.SessionFluorophore & 'fluorophore!="Twitch2B"'); 
	end

	methods(Access=protected)

		function makeTuples(self, key)
            % Copy traces from ExtractRawTrace
            tuples = fetch(preprocess.ExtractRawTrace & key,'raw_trace->trace','*');
            tuples = rmfield(tuples,'channel');
            
            self.insert(key)
            insert(preprocess.ComputeTracesTrace,tuples);
		end
	end

end


