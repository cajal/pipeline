%{
preprocess.SpikesRateTrace (computed) # Inferred spikes
-> preprocess.Spikes
-> preprocess.ComputeTracesTrace
---
rate_trace=null             : longblob                      # leave null if same as ExtractRaw.Trace
%}


classdef SpikesRateTrace < dj.Relvar
    
    methods
        
        function makeTuples(self, key)
            method = fetch1(preprocess.SpikeMethod & key, 'spike_method_name') ;
            
            switch method    
                case 'nmf'
                    % Copy NMF spikes from preprocess.ExtractRawSpikeRate
                    keys =  rmfield(...
                        fetch(preprocess.ExtractRawSpikeRate & key, 'spike_trace->rate_trace', '*'), ...
                        'channel');
                    [keys.spike_method]=deal(key.spike_method);
                    self.insert(keys)
                    
                case 'improved oopsi'
                    
                    % get stuff
                    fps = fetch1(preprocess.PrepareGalvo & key,'fps');
                    [traces, keys] = fetchn(preprocess.ComputeTracesTrace ...
                        & key ,'trace');
                    traces = double(cell2mat(traces'));
                    
                    % remove 1PC
                    [c, p] = pca(traces);
                    traces = p(:,2:end)*c(:,2:end)';
    
                    % high pass filter
                    hp = 0.02; 
                    traces = traces + abs(min(traces(:)))+eps;
                    traces = traces./ne7.dsp.convmirr(traces,hamming(round(fps/hp)*2+1)/sum(hamming(round(fps/hp)*2+1)))-1;  %  dF/F where F is low pass
                    traces = bsxfun(@plus,traces,abs(min(traces)))+eps;

                    % fast oopsi
                    parfor iTrace = 1:size(traces,2)
                    	keys(iTrace).rate_trace = fast_oopsi(traces(:,iTrace)', struct('dt',1/fps),struct('lambda',.2));
                    end
                    
                    % insert
                    [keys.spike_method]=deal(key.spike_method);
                    self.insert(keys)
                    
                otherwise
                    error('method "%s" not implemented', method)
                    
            end
        end
        
    end
end