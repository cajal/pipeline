%{
rf.GratingResponses (computed) # calcium responses for each trace and grating trials
-> rf.Sync
-> rf.Trace
-> rf.SpaceTime
---
spike_responses        : longblob                      # column vector of calcium responses per trial (nans are possible for unbalanced experiments)
%}

classdef GratingResponses < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = rf.Sync*rf.Segment & psy.Grating
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            disp 'fetching trials and traces ...'
            latency = 0.01;  % delay for integration window
            caTimes = fetch1(rf.Sync & key, 'frame_times') - latency;
            dt = median(diff(caTimes));
            attrs = {'flip_times', 'direction', 'spatial_freq', 'temp_freq'};
            q = fetch(rf.Sync*psy.Trial*psy.Grating & key & 'trial_idx between first_trial and last_trial', attrs{:});
            [flipTimes, directions, spatialFreqs, tempFreqs] = dj.struct.tabulate(q, attrs{:});
            [directions, order] = sort(directions);
            flipTimes = flipTimes(order,:,:,:);
            [X,traceKeys] = fetchn(rf.Trace & key, 'ca_trace');
            X = [X{:}];
            if size(X,1)>length(caTimes)
                warning 'patch stim labview software was aborted.'
                X = X(1:length(caTimes),:);
            end
            if size(X,2)<length(caTimes)
                warning 'scan aborted'
                caTimes = caTimes(1:size(X,1));
            end
            assert(size(X,1)==length(caTimes), 'calcium times and trace lengths do not match')
            
            disp 'deconvolving...'
            
            % high-pass filter
            cutoff = 0.05;
            k = hamming(round(1/dt/cutoff)*2+1);
            k = k/sum(k);
            m = mean(X);
            X = bsxfun(@rdivide,X-ne7.dsp.convmirr(double(X),k),m);
            
            % fast oopsi
            for i=1:size(X,2)
                X(:,i) = fast_oopsi(double(X(:,i)),struct('dt',dt),struct('lambda',0.3));
            end
            
            disp 'averaging responses...'
            
            responses = cellfun(@(t) xsum(X,caTimes,t), flipTimes, 'uni', false);
            
            disp 'inserting...'
            for iSpatial = [-1 1:length(spatialFreqs)]
                for iTemp = [-1 1:length(tempFreqs)]
                    spaceTimeKey = struct;
                    if iSpatial == -1
                        spaceTimeKey.spatial_freq = -1;
                        ixSpatial = 1:length(spatialFreqs);
                    else
                        spaceTimeKey.spatial_freq = spatialFreqs(iSpatial);
                        ixSpatial = iSpatial;
                    end
                    if iTemp == -1
                        spaceTimeKey.temp_freq = -1;
                        ixTemp = 1:length(tempFreqs);
                    else
                        spaceTimeKey.temp_freq = tempFreqs(iTemp);
                        ixTemp = iTemp;
                    end
                    inserti(rf.SpaceTime,spaceTimeKey)
                    r = reshape(responses(:,ixSpatial,ixTemp,:), length(directions), []);
                    for iTrace = 1:length(traceKeys)
                        tuple = dj.struct.join(key,spaceTimeKey);
                        tuple = dj.struct.join(tuple,traceKeys(iTrace));
                        tuple.spike_responses = cellfun(@(r) r(iTrace), r);
                        self.insert(tuple)
                    end
                end
            end
        end
    end
    
end



function ret = xsum(X,caTimes,flipTimes)
if isempty(flipTimes)
    ret = nan(1,size(X,2),'single');
else
    ret = sum(X(caTimes>flipTimes(2) & caTimes<flipTimes(end),:));
end
end