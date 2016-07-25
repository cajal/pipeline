%{
aod_monet.VonMises (computed) # directional tuning
-> aod_monet.DriftResponseSet
-> aodpre.Spikes
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
        popRel  = aod_monet.DriftResponseSet 
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            s = fetch(aod_monet.DriftTrial*aod_monet.DriftResponse & key, 'direction','response');
            [responses, traceIds, direction] = dj.struct.tabulate(s,...
                'response', 'point_id', 'direction');
            assert(all(diff(direction)>0), 'just a check that directions are monotonic')
            nShuffles = 1e4;
            [von, r2, p] = ne7.rf.VonMises2.computeSignificance(double(responses), nShuffles);
            r2(isnan(r2)) = 0;
            for iTrace=1:length(traceIds)
                tuple = key;
                tuple.point_id = traceIds(iTrace);
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