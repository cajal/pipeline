%{
rf.Sync (imported) # mapping of h5,ca imaging, and vis stim clocks
-> rf.Scan
---
-> psy.Session
first_trial                 : int                           # first trial in recording
last_trial                  : int                           # last trial in recording
frame_times                 : longblob                      # times of frames and slices
sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
%}

classdef Sync < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = rf.Scan
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            assert(numel(key)==1)
            tuple = key;
            % find frame pulses
            dat = rf.readHD5(key);
            packetLen = 2000;
            if isfield(dat,'analogPacketLen')
                packetLen = dat.analogPacketLen;
            end
            datT = pipetools.ts2sec(dat.ts, packetLen);
            dt = median(diff(datT));
            fs=1/dt;
            n = ceil(0.0002/dt);
            k = hamming(2*n);
            k = -k/sum(k);
            k(1:n) = -k(1:n);
            
%%
            % detect flips in the recorded photodiode signal
            [detectedFlipInd, detectedFlipNums] = ne7.dsp.FlipCode.whichFlips(dat.syncPd, fs);
            
            % fix for files after feb 15th 2013
            detectedFlipNums = detectedFlipNums-1;
            
            % consider all trials with flip nums within the min and max of detected flip nums
            trials = psy.Trial & key & sprintf('last_flip_count between %d and %d', min(detectedFlipNums), max(detectedFlipNums));
            
            % get all flip times for each trial, also get the number of the last flip in the trial
            [psyId, trialLastFlipNum,trialFlipTimes] = fetchn(trials,'psy_id', 'last_flip_count', 'flip_times');
            
            % check to make sure a new psy session was not started in the middle of the recording
            % this is not allowed so that condition parameters to remain constant within each psy session
            assert(all(psyId == psyId(1)), 'Multiple PsySessions in scan: not allowed.');
            psyId = psyId(1);
            
            % sort the trials in order
            [trialLastFlipNum, order] = sort(trialLastFlipNum);
            trialFlipTimes = trialFlipTimes(order);
            
            % fill in the flip numbers within each trial (counting up to the last flip in the trial)
            trialFlipNums = [];
            for i=1:length(trialLastFlipNum)
                trialFlipNums = [trialFlipNums  trialLastFlipNum(i)+(1-length(trialFlipTimes{i}):0)]; %#ok<AGROW>
            end
            trialFlipTimes = [trialFlipTimes{:}];
            
            % now every recorded flip time should have a flip number
            assert(length(trialFlipTimes)==length(trialFlipNums));
            
            % find the common flip numbers between the detected and recorded flips
            commonFlipNums = intersect(detectedFlipNums, trialFlipNums);
            
            % make sure we have a decent amount of matched flip numbers
            if length(commonFlipNums)<100
                warning('Insufficient matched flips (%d), skipping...',length(commonFlipNums))
                key
                return
            end
            
            % get the index into the photodiode signal for all the matched flips
            pDiodeFlipInd = detectedFlipInd(ismember(detectedFlipNums,commonFlipNums));
            
            % get the visual stim system time for all the matched flips
            visFlipTime = trialFlipTimes(ismember(trialFlipNums,commonFlipNums));
            
            % regress the photodiode indices against the vis stim times to get the photodiode signal time on the mac
            % assumes uninterrupted uniform sampling of photodiode!!!
            pDiodeFlipTime=datT(pDiodeFlipInd)';
            mx = mean(pDiodeFlipTime);
            b = robustfit(pDiodeFlipTime-mx, visFlipTime);
            visTime = (datT'-mx)*b(2)+b(1);
            
            tuple.psy_id = psyId;
            
            % get all the trials that are fully within visTime(1:end)
            [trialIds, flipTimes] = fetchn(psy.Trial & tuple, 'trial_idx', 'flip_times');
            ix = cellfun(@(x) any(x>=visTime(1) & x<=visTime(end)), flipTimes);
            trialIds = trialIds(ix);
                        
            % store the first and last trial
            tuple.first_trial = trialIds(1);
            tuple.last_trial = trialIds(end);
            
            %figure;plot(visFlipTime - visTime(pDiodeFlipInd))
            if quantile(abs(visFlipTime - visTime(pDiodeFlipInd)),0.9) > .01
                warning('Incorrectly detected flips (%f), skipping...',...
                    quantile(abs(visFlipTime - visTime(pDiodeFlipInd)),0.9))
                disp(key)
                return
            end
            
            pulses = conv(dat.scanImage,k,'same');
            peaks = ne7.dsp.spaced_max(pulses, 0.005/dt);
            peaks = peaks(pulses(peaks) > 0.1*quantile(pulses(peaks),0.9));
            peaks = longestContiguousBlock(peaks);                        
            tuple.frame_times = visTime(peaks);
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
