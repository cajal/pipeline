%{
monet.DriftResponse (computed) # calcium responses to drift trials
-> monet.DriftResponseSet
-> monet.DriftTrial
-> pre.Trace
-----
response: float  # averaged response
%}

classdef DriftResponse < dj.Relvar

	methods

		function makeTuples(self, key)
            disp 'preparing traces...'
            frameTimes = fetch1(rf.Sync & key, 'frame_times');
            [traces, traceKeys] = fetchn(pre.Spikes & key, 'spike_trace');
            traces = [traces{:}];
            nslices = fetch1(pre.ScanInfo & key, 'nslices');
            frameTimes = frameTimes(key.slice:nslices:end);
            
            disp 'snipping...'
            latency = 0.03;
            trials = fetch(monet.DriftTrial & key, 'onset', 'offset', 'ORDER BY drift_trial');
            for trial = trials'
                if mod(trial.drift_trial, 20)==0 || trial.drift_trial==trials(end).drift_trial
                    fprintf('Trial %3d/%d\n', trial.drift_trial, trials(end).drift_trial)
                end
                responses = num2cell(mean(traces(frameTimes > trial.onset+latency & frameTimes < trial.offset+latency,:)));
                tuples = dj.struct.join(traceKeys, rmfield(trial, {'onset', 'offset'}));
                [tuples.response] = deal(responses{:});
                self.insert(tuples)
            end
            
            disp done
		end
    end

    
end
