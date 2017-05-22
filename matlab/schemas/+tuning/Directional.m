%{
# all directional drift trials for the scan
-> preprocess.Sync
---
ndirections                 : tinyint                       # number of directions
%}


classdef Directional < dj.Computed

    properties
        keySource = preprocess.Sync  & (vis.Trial*vis.Monet & 'speed>0' & 'trial_idx between first_trial and last_trial');
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            iTrial = 0;
            trialTuples = [];
            for trialKey = fetch(vis.Trial * preprocess.Sync & vis.Monet & key & 'trial_idx between first_trial and last_trial')'
                frameTimes = fetch1(preprocess.Sync & trialKey, 'frame_times');
                [params, flips] = fetch1(...
                    vis.Monet*vis.MonetLookup*vis.Trial & trialKey, ...
                    'params', 'flip_times');
                frametimes = params{4}.frametimes;
                directions = params{4}.direction;
                onsets = interp1(frametimes, flips, params{4}.onsets);
                offsets = interp1(frametimes, flips, params{4}.offsets);
                assert(~any(isnan(onsets)) && ~any(isnan(offsets)), 'invalid trial times')
                assert(length(flips)==length(params{4}.frametimes) ...
                    && max(abs(diff(frametimes(:))-diff(flips(:)))), 'invalid frames')
                for i=1:length(onsets)
                    if onsets(i) > frameTimes(1)+0.5 && offsets(i) < frameTimes(end)-0.5
                        iTrial = iTrial + 1;
                        tuple = trialKey;
                        tuple.drift_trial = iTrial;
                        tuple.onset = onsets(i);
                        tuple.offset = offsets(i);
                        tuple.direction = directions(i)/pi*180;
                        trialTuples = cat(1,trialTuples, tuple);
                    end
                end
            end
            directions = unique([trialTuples.direction]);
            key.ndirections = length(directions);
            assert(key.ndirections>=8 && all(diff(diff(directions)) < 1e-7), 'directions must be uniform')
            self.insert(key)
            insert(tuning.DirectionalTrial, trialTuples)
        end
        
    end

end
