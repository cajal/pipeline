%{
preprocess.Treadmill (imported) # 
-> experiment.Scan
---
treadmill_raw                        : longblob          # raw ball velocity
treadmill_vel                        : longblob          # ball velocity integrated over 100ms bins in cm/sec
treadmill_time                      : longblob          # timestamps of each sample in seconds, with same t=0 as patch and camera data
treadmill_cmsec_factor               : float             # factor used to convert to cm/2
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
            
            bins100ms = ballT(1):.1:ballT(end);
            distPerBin = [0 diff(interp1(ballT,tuple.treadmill_raw-tuple.treadmill_raw(1),bins100ms))]; % integrated distance per 100ms bin
            
            tuple.treadmill_cmsec_factor = .09; %(71.8168 cm circumference of 9inch wheel generates 8000 encoder counts = ~.009cm per count * 10 100ms bins per second)
            tuple.treadmill_vel = interp1(bins100ms,distPerBin * tuple.treadmill_cmsec_factor,ballT);
            tuple.treadmill_vel(isnan(tuple.treadmill_vel))=0;
           
            self.insert(tuple)
        end
    end
    
end