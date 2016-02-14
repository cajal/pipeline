%{
aodpre.Spikes (computed) # traces of infered firing rates
-> aodpre.ExtractSpikes
-> aodpre.Trace
-----
spike_trace :longblob
%}

classdef Spikes < dj.Relvar
    
    methods      
        
        function makeTuples(self, key)
            dt = 1/fetch1(aodpre.Scan & key, 'sampling_rate');
            [X, traceKeys] = fetchn(aodpre.Trace & key, 'trace');
            X = infer_spikes(pre.SpikeInference & key, cat(2,X{:}), dt);
            for i=1:length(traceKeys)
                tuple = dj.struct.join(key, traceKeys(i));
                tuple.spike_trace = X(:,i);
                self.insert(tuple)
            end
        end
    end
    
end