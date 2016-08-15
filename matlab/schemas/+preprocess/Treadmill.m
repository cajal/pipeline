%{
preprocess.Treadmill (imported) # 
-> experiment.Scan
---
treadmill_raw                        : longblob          # raw treadmill counts
treadmill_vel                        : longblob          # ball velocity integrated over 100ms bins in cm/sec
treadmill_time                       : longblob           # timestamps of each sample in seconds on behavior clock
treadmill_ts=CURRENT_TIMESTAMP       : timestamp         # automatic
%}


classdef Treadmill < dj.Relvar & dj.AutoPopulate
    
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
            ballT = pipetools.ts2sec(dat.ball(:,2), packetLen);
            tuple.treadmill_time = ballT;
            
            tuple.treadmill_raw = unwrap(dat.ball(:,1),2^31);
            
            % Calculate counts per bin
            bins100ms = ballT(1):.1:ballT(end);
            countsPerBin = [0 diff(interp1(ballT,tuple.treadmill_raw-tuple.treadmill_raw(1),bins100ms))]; % integrated (+ and -) counts per 100ms bin
            
            % Get all prior treadmill spec keys
            treadmillSpecKeys = fetch(experiment.TreadmillSpecs * experiment.Session & key & 'treadmill_start_date <= session_date','ORDER BY treadmill_start_date');
            % Get diameter and encoder counts per revolution for most recent treadmill
            [diam, counts] = fetch1(experiment.TreadmillSpecs & treadmillSpecKeys(end),'diameter','counts_per_revolution');
            
            cmPerCount = pi*diam/counts; % for example, 71.8168 cm circumference of 9inch wheel generates 8000 encoder counts = ~.009cm per count
 
            tuple.treadmill_vel = interp1(bins100ms,countsPerBin * cmPerCount * 10,ballT); % multiply counts per bin * cm per count * ten 100ms bins per second to get cm/s
            tuple.treadmill_vel(isnan(tuple.treadmill_vel))=0;
           
            self.insert(tuple)
        end
    end
    
end