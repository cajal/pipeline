%{
aod_monet.DriftResponse (computed) # calcium responses to drift trials
-> aod_monet.DriftResponseSet
-> aod_monet.DriftTrial
-> aodpre.Trace
-----
response: float  # averaged response
%}

classdef DriftResponse < dj.Relvar

	methods

		function makeTuples(self, key)
            disp 'preparing traces...'
            [start_time, duration] = fetch1(aodpre.Sync & key, 'signal_start_time', 'signal_duration');
            [traces, traceKeys] = fetchn(aodpre.Spikes & key, 'spike_trace');
            traces = [traces{:}];
            frameTimes = start_time + linspace(0, duration, size(traces,1));
            
            disp 'snipping...'
            latency = 0.03;
            for trialKey = fetch(aod_monet.DriftTrial & key, 'onset', 'offset')'
                responses = num2cell(mean(traces(frameTimes > trialKey.onset+latency & frameTimes < trialKey.offset+latency,:)));
                tuples = dj.struct.join(traceKeys, rmfield(trialKey, {'onset', 'offset'}));
                [tuples.response] = deal(responses{:});
                self.insert(tuples)
            end
            
            disp done
		end
    end

end
