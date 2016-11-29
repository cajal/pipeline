%{
preprocess.SpikesRateTrace (computed) # Inferred spikes
-> preprocess.Spikes
-> preprocess.ComputeTracesTrace
---
rate_trace=null             : longblob                      # leave null if same as ExtractRaw.Trace
%}


classdef SpikesRateTrace < dj.Relvar
    
    methods
        
        function makeTuples(self, key)
            method = fetch1(preprocess.SpikeMethod & key, 'spike_method_name') ;
            
            switch method    
                case 'nmf'
                    % Copy NMF spikes from preprocess.ExtractRawSpikeRate
                    keys =  rmfield(...
                        fetch(preprocess.ExtractRawSpikeRate & key, 'spike_trace->rate_trace', '*'), ...
                        'channel');
                    [keys.spike_method]=deal(key.spike_method);
                    self.insert(keys)
                                        
                otherwise
                    error('invalid method %s', method)
                    
            end
        end
        
    end
end