%{
aodpre.ComputeTraces (computed) # traces used for spike extraction
-> aodpre.Set
-----
%}

classdef ComputeTraces < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = aodpre.Set;
	end

	methods(Access=protected)

		function makeTuples(self, key)
			self.insert(key)
            %for regular imaging, use channel 1.  Handle ratiometric separately
            insert(aodpre.Trace, rmfield(fetch(aodpre.Timeseries & key & 'channel=1', '*'), 'channel'))
		end
	end

end