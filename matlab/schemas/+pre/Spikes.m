%{
pre.Spikes (computed) # traces of infered firing rates
-> pre.ExtractSpikes
-> pre.Trace
-----
spike_trace :longblob
%}

classdef Spikes < dj.Relvar
    
    methods
        function plot(self)
            for key = fetch(pre.Segment & self)'
                X = fetchn(self & key, 'spike_trace');
                X = [X{:}];
                t = fetch1(rf.Sync & key, 'frame_times');
                X = bsxfun(@plus,bsxfun(@rdivide,X,mean(X))/40,1:size(X,2));
                nslices = fetch1(pre.StackInfo & key, 'nslices');
                plot(t(1:nslices:end)-t(1),X)
            end
        end
        
        
        function makeTuples(self, key)
            dt = 1/fetch1(pre.ScanInfo & key, 'fps');
            [X, traceKeys] = fetchn(pre.Trace & key, 'ca_trace');
            X = infer_spikes(pre.SpikeInference & key, cat(2,X{:}), dt);
            for i=1:length(traceKeys)
                tuple = dj.struct.join(key, traceKeys(i));
                tuple.spike_trace = X(:,i);
                self.insert(tuple)
            end
        end
    end
    
end