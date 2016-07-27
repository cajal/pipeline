%{
preprocess.ComputeTracesTrace (computed) # final calcium trace but before spike extraction or filtering
-> preprocess.ComputeTraces
-> preprocess.ExtractRawTrace
---
trace=null                  : longblob                      # leave null same as ExtractRaw.Trace
%}


classdef ComputeTracesTrace < dj.Relvar
	methods

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end