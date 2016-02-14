%{
pre.ExtractTraces (imported) # my 
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