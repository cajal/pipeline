classdef STMSpikeRate < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = 0
	end

	methods(Access=protected)

		function makeTuples(self, key)
            error('This table is populated from python.');
		end
	end

end