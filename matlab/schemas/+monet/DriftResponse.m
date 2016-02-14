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
            
            disp 'snipping...'
            latency = 0.03;
            for trialKey = fetch(monet.DriftTrial & key)'
                [onset, offset] = fetch1(monet.DriftTrial & trialKey, 'onset', 'offset');
                responses = num2cell(mean(traces(frameTimes > onset+latency & frameTimes < offset+latency,:)));
                tuples = dj.struct.join(traceKeys, trialKey);
                [tuples.response] = deal(responses{:});
                self.insert(tuples)
            end
            
            disp done
		end
    end

end
