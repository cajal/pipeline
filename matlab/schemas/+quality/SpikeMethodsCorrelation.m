%{
# 
-> quality.SpikeMethods
-> preprocess.ComputeTracesTrace
---
corr=null                   : float                         # correlation between spike method 1 and 2 on that trace
%}


classdef SpikeMethodsCorrelation < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end