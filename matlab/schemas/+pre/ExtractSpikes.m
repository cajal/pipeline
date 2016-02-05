%{
pre.ExtractSpikes (computed) #  inferences of spikes from calcium traces
-> pre.Segment
-> pre.SpikeInference
-----
%}

classdef ExtractSpikes < dj.Relvar & dj.AutoPopulate

	properties
		popRel = pre.Segment * pre.SpikeInference & rf.Sync & struct('language','matlab')
	end

	methods(Access=protected)

		function makeTuples(self, key)
			self.insert(key)
            makeTuples(pre.Spikes, key)
		end
	end

end