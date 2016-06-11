%{
monet.DriftTrialSet (computed) # all drift trials for this scan
-> rf.Sync
---
%}


classdef DriftTrialSet < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = rf.Sync  & (psy.MovingNoise & 'speed>0');
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            self.insert(key)
            iTrial = 0;
            for key = fetch(psy.Trial * rf.Sync & psy.MovingNoise & key)'
                iTrial = makeTuples(monet.DriftTrial, key, iTrial);
            end
        end
        
    end
end