%{
monet.VonMises (computed) # directional tuning
-> monet.DriftResponseSet
-> pre.Spikes
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
        popRel  = monet.DriftResponseSet
    end
    
    
    methods
        function plot(self)
            for key = fetch(monet.DriftResponseSet & self)'
                figure
                s = fetch(monet.DriftTrial*monet.DriftResponse & self & key, 'direction','response');
                [responses, traceIds, direction] = dj.struct.tabulate(s,...
                    'response', 'mask_id', 'direction');
                ncells = length(traceIds);
                ncols = ceil(sqrt(ncells));
                nrows = ceil(ncells/ncols);
                angles = (0:size(responses,2)-1)/size(responses,2)*360;
                for icell=1:ncells
                    subplot(nrows, ncols, icell)
                    plot(angles, squeeze(responses(icell, :, :))', 'k.')
                    hold on
                    r = squeeze(responses(icell,:,:));
                    m = mean(r,2);
                    s = std(r,[], 2)./sqrt(sum(~isnan(r),2));
                    errorbar(angles, m, s, 'r', 'LineWidth', 3)
                    ylim([0 max(m)*4])
                    xlim([0 360])
                    box off
                end
            end
        end
    end
    
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            s = fetch(monet.DriftTrial*monet.DriftResponse & key, 'direction','response');
            [responses, traceIds, direction] = dj.struct.tabulate(s,...
                'response', 'mask_id', 'direction');
            assert(all(diff(direction)>0), 'just a check that directions are monotonic')
            nShuffles = 1e4;
            [von, r2, p] = ne7.rf.VonMises2.computeSignificance(double(responses), nShuffles);
            r2(isnan(r2)) = 0;
            for iTrace=1:length(traceIds)
                tuple = key;
                tuple.mask_id = traceIds(iTrace);
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