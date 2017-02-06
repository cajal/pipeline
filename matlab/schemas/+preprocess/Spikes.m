%{
preprocess.Spikes (computed) # infer spikes from calcium traces
-> preprocess.ComputeTraces
-> preprocess.SpikeMethod
---
%}


classdef Spikes < dj.Relvar & dj.AutoPopulate
    
    properties
        % NMF spikes (method 5) can be copied via MATLAB, STM spike detection is performed in Python
        popRel = preprocess.ComputeTraces * preprocess.SpikeMethod & 'language="matlab"';
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            if key.spike_method == 5 && key.extract_method ~= 2
                disp 'NMF spikes exist only under NMF extraction!'
                disp 'skipping'
                return
            else
                self.insert(key)
                makeTuples(preprocess.SpikesRateTrace, key)
            end

        end
    end
end