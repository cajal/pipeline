%{
preprocess.BehaviorSync (imported) # 
-> experiment.Scan
---
frame_times=null                    : longblob                      # times of frames and slices on behavior clock
behavior_sync_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
%}


classdef BehaviorSync < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = experiment.Scan
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            assert(numel(key)==1,'One key at a time, please.')
            tuple=key;
            % read photodiode signal
            dat = preprocess.readHD5(key);
            packetLen = 2000;
            if isfield(dat,'analogPacketLen')
                packetLen = dat.analogPacketLen;
            end
            datT = pipetools.ts2sec(dat.ts, packetLen);
            dat_fs = 1/median(diff(datT));
            
            % find scanimage frame pulses
            n = ceil(0.0002*dat_fs);
            k = hamming(2*n);
            k = -k/sum(k);
            k(1:n) = -k(1:n);
            pulses = conv(dat.scanImage,k,'same');
            peaks = ne7.dsp.spaced_max(pulses, 0.005*dat_fs);
            peaks = peaks(pulses(peaks) > 0.1*quantile(pulses(peaks),0.9));
            peaks = longestContiguousBlock(peaks);
            tuple.frame_times = datT(peaks);
           
            self.insert(tuple)
        end
    end
    
end


function idx = longestContiguousBlock(idx)
d = diff(idx);
ix = [0 find(d > 10*median(d)) length(idx)];
f = cell(length(ix)-1,1);
for i = 1:length(ix)-1
    f{i} = idx(ix(i)+1:ix(i+1));
end
l = cellfun(@length, f);
[~,j] = max(l);
idx = f{j};
end
