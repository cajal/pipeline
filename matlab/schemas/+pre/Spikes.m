%{
pre.Spikes (computed) # traces of infered firing rates
-> pre.ExtractSpikes
-> pre.Trace
-----
spike_trace :longblob  
%}

classdef Spikes < dj.Relvar 

	methods

		function makeTuples(self, key)
            times = fetch1(rf.Sync & key, 'frame_times');
            dt = median(diff(times));
            [X, traceKeys] = fetchn(pre.Trace & key, 'ca_trace');
            X = infer_spikes(pre.SpikeInference & key, cat(2,X{:}), dt);
            for i=1:length(traceKeys)
                tuple = dj.struct.join(key, traceKeys(i));
                tuple.spike_trace = X(:,i);
                self.insert(tuple)
            end
		end
	end

end