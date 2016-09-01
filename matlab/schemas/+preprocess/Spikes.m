%{
preprocess.Spikes (computed) # infer spikes from calcium traces
-> preprocess.ComputeTraces
-> preprocess.SpikeMethod
---
%}


classdef Spikes < dj.Relvar & dj.AutoPopulate
    
    properties
        % NMF spikes (method 5) can be copied via MATLAB, STM spike detection is performed in Python
        popRel = preprocess.ComputeTraces*preprocess.SpikeMethod  & preprocess.ExtractRawSpikeRate & 'spike_method=5';
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            % Copy NMF spikes from preprocess.ExtractRawSpikeRate
            tuples = fetch(preprocess.ExtractRawSpikeRate & key,'spike_trace->rate_trace','*');
            tuples = rmfield(tuples,'channel');
            [tuples.spike_method] = deal(5);
            
            self.insert(key)
            insert(preprocess.SpikesRateTrace,tuples);
            
            
        end
    end
    
    
    methods
        
        function NewTraces = getTraces(self) % Adjusts traces for time difference between slices in a scan
            
            [Traces, slice] = fetchn( ...
                preprocess.SpikesRateTrace * preprocess.ExtractRawGalvoROI ...
                & self, 'rate_trace', 'slice' );
            nslices = length(unique(slice));
            Traces = [Traces{:}];
            CaTimes = 1:size(Traces,1)*nslices;
            NewTraces = nan(size(Traces));
            NewTimes = CaTimes(1:nslices:end);
            
            for islice = 1:nslices
                caTimes = CaTimes(islice:nslices:end);
                X = Traces(:,islice==slice);
                xm = min([length(caTimes) length(X)]);
                X = @(t) interp1(caTimes(1:xm), X(1:xm,:), t, 'linear', 'extrap');  % traces indexed by time
                
                NewTraces(:,islice==slice) = X(NewTimes);
            end
        end
    end
end