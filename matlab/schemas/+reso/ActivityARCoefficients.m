%{
# fitted parameters for the autoregressive process (nmf deconvolution)
-> reso.ActivityTrace
---
g                           : blob                          # g1, g2, ... coefficients for the AR process
%}


classdef ActivityARCoefficients < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end