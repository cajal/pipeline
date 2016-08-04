%{
tuning.DirectionalResponseTrial (computed) # the response for each trial and each trace
-> tuning.DirectionalResponse
-> preprocess.SpikesRateTrace
-> tuning.DirectionalTrial
---
response                    : float                         # integrated response
%}


classdef DirectionalResponseTrial < dj.Relvar
	methods

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end