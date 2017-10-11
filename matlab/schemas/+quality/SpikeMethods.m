%{
# 
-> preprocess.ComputeTraces
 (spike_method_1) -> preprocess.Spikes
 (spike_method_2) -> preprocess.Spikes
%}


classdef SpikeMethods < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end