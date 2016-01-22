%{
rf.VonMises (computed) # orientation and spatiotemporal tuning
-> rf.GratingResponses
-----
von_r2     : float  # r-squared explaned by vonMises fit
von_pref   : float  #  preferred directions
von_base   : float  #  von mises base value
von_amp1   : float  #  amplitude of first peak
von_amp2   : float  #  amplitude of second peak
von_sharp  : float  #  sharpnesses
von_pvalue : float  # p-value by shuffling (nShuffles = 1e4)
%}

classdef VonMises < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = rf.Sync*rf.SpaceTime & rf.GratingResponses
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            [responses, traceKeys] = fetchn(rf.GratingResponses & key, 'spike_responses');
            responses = permute(double(cat(3,responses{:})),[3 1 2]);
            
            nShuffles = 1e4;
            [von, r2, p] = ne7.rf.VonMises2.computeSignificance(responses, nShuffles);
            r2(isnan(r2)) = 0;
            for iTrace=1:length(traceKeys)
                tuple = traceKeys(iTrace);
                tuple.von_r2 = r2(iTrace);
                tuple.von_base = von.w(iTrace,1);
                tuple.von_amp1 = von.w(iTrace,2);
                tuple.von_amp2 = von.w(iTrace,3);
                tuple.von_sharp= von.w(iTrace,4);
                tuple.von_pref = von.w(iTrace,5);
                tuple.von_pvalue = p(iTrace);
                self.insert(tuple)
            end
        end
        
    end
end