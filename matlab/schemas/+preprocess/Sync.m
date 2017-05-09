%{
# produces synchronization information between two-photon scanning and visual stimulus
-> preprocess.Prepare
---
-> vis.Session
first_trial                 : int                           # first trial index from psy.Trial overlapping recording
last_trial                  : int                           # last trial index from psy.Trial overlapping recording
signal_start_time           : double                        # (s) signal start time on stimulus clock
signal_duration             : double                        # (s) signal duration on stimulus time
frame_times=null            : longblob                      # times of frames and slices
sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
%}


classdef Sync < dj.Imported
    
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            assert(numel(key)==1)
            
            % read photodiode signal
            dat = preprocess.readHD5(key);
            packetLen = 2000;
            if isfield(dat,'analogPacketLen')
                packetLen = dat.analogPacketLen;
            end
            datT = pipetools.ts2sec(dat.ts, packetLen);
            
            photodiode_fs = 1/median(diff(datT));
            photodiode_signal = dat.syncPd;
            fps = 60;   % does not need to be exact
            
            % synchronize to stimulus
            tuple =  stims.analysis.sync(key, photodiode_signal, photodiode_fs, fps, vis.Trial);
            
            % find scanimage frame pulses
            n = ceil(0.0002*photodiode_fs);
            k = hamming(2*n);
            k = -k/sum(k);
            k(1:n) = -k(1:n);
            pulses = conv(dat.scanImage,k,'same');
            peaks = ne7.dsp.spaced_max(pulses, 0.005*photodiode_fs);
            peaks = peaks(pulses(peaks) > 0.1*quantile(pulses(peaks),0.9));
            peaks = longestContiguousBlock(peaks);
            tuple.frame_times = tuple.signal_start_time + tuple.signal_duration*(peaks-1)/length(dat.scanImage);
            
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
