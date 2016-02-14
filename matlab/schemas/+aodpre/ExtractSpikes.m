%{
aodpre.ExtractSpikes (computed) # my newest table
-> aodpre.Scan
-> pre.SpikeInference
-----
# add additional attributes
%}

classdef ExtractSpikes < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = aodpre.Scan * pre.SpikeInference & 'language="matlab"'
	end

	methods(Access=protected)

		function makeTuples(self, key)
            self.insert(key)
            makeTuples(aodpre.Spikes, key)
		end
	end

end