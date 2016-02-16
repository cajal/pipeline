%{
aod_monet.DriftTrialSet (computed) # all drift trials for this scan
-> aodpre.Scan
---
%}


classdef DriftTrialSet < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = aodpre.Scan  & (psy.MovingNoise & 'speed>0');
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            self.insert(key)
            iTrial = 0;
            for key = fetch(psy.Trial * aodpre.Scan & key)'
                iTrial = makeTuples(aod_monet.DriftTrial, key, iTrial);
            end
        end
        
    end
end