%{
monet.DriftResponseSet (computed) # my newest table
-> monet.DriftTrialSet
-> pre.ExtractSpikes
---
%}


classdef DriftResponseSet < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = monet.DriftTrialSet*pre.Segment*pre.ExtractSpikes & pre.Spikes
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            self.insert(key)
            makeTuples(monet.DriftResponse, key)
        end
    end
    
end
