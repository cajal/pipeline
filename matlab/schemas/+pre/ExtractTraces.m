%{
pre.ExtractTraces (imported) # my newest table
-> pre.Segment
-----
%}

classdef ExtractTraces < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = pre.Segment
	end

	methods(Access=protected)

		function makeTuples(self, key)
			self.insert(key)
            makeTuples(pre.Trace, key)
		end
	end

end