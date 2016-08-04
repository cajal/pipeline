%{
tuning.DirectionalResponse (computed) # response to directional stimulus
-> tuning.Directional
-> preprocess.Spikes
---
latency                     : float                         # latency used (ms)
%}


classdef DirectionalResponse < dj.Relvar & dj.AutoPopulate

	properties
		popRel = tuning.Directional*preprocess.Spikes 
	end

	methods(Access=protected)

		function makeTuples(self, key)
            error 'This module is implemented in python'
		end
	end

end