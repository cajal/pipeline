%{
preprocess.Spikes (computed) # infer spikes from calcium traces
-> preprocess.ComputeTraces
-> preprocess.SpikeMethod
---
%}


classdef Spikes < dj.Relvar & dj.AutoPopulate
    
    properties
        % NMF spikes (method 5) can be copied via MATLAB, STM spike detection is performed in Python
        popRel = preprocess.ComputeTraces*preprocess.SpikeMethod  & preprocess.ExtractRawSpikeRate & 'language="matlab"';
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            self.insert(key)
            makeTuples(preprocess.SpikesRateTrace, key)

        end
    end
end