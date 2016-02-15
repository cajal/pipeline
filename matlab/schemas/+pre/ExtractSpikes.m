%{
pre.ExtractSpikes (computed) # inferences of spikes from calcium traces
-> pre.ExtractTraces
-> pre.SpikeInference
---
%}

classdef ExtractSpikes < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = pre.ExtractTraces * pre.SpikeInference & struct('language','matlab')
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            self.insert(key)
            makeTuples(pre.Spikes, key)
        end
    end
    
end
