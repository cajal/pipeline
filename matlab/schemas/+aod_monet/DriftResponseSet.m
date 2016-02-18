%{
aod_monet.DriftResponseSet (computed) # my newest table
-> aod_monet.DriftTrialSet
-> aodpre.ExtractSpikes
---
%}


classdef DriftResponseSet < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = aod_monet.DriftTrialSet*aodpre.ExtractSpikes & aodpre.Spikes
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            self.insert(key)
            makeTuples(aod_monet.DriftResponse, key)
        end
    end
    
end
