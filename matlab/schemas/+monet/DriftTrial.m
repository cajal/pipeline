%{
monet.DriftTrial (computed) # noise drift trials
-> monet.DriftTrialSet
drift_trial     : smallint               # trial index
---
-> psy.Trial
direction                   : float                         # (degrees) direction of drift
onset                       : double                        # (s) onset time in rf.Sync times
offset                      : double                        # (s) offset time in rf.Sync times
%}


classdef DriftTrial < dj.Relvar
    methods
        
        function iTrial = makeTuples(self, key, iTrial)
            tuple = key;
            tuple.drift_trial = 0;
            frameTimes = fetch1(rf.Sync & key, 'frame_times');
            [params, flips] = fetch1(psy.MovingNoise*psy.MovingNoiseLookup*psy.Trial & key, ...
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
                    tuple.drift_trial = iTrial;
                    tuple.onset = onsets(i);
                    tuple.offset = offsets(i);
                    tuple.direction = directions(i);
                    self.insert(tuple)
                end
            end
        end
    end
end