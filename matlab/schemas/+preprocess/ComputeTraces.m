%{
preprocess.ComputeTraces (computed) # compute traces
-> preprocess.ExtractRaw
---
%}


classdef ComputeTraces < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.ExtractRaw  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
            error('Implemented in python.')
		end
	end

end


