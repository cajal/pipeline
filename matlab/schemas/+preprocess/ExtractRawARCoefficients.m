%{
# Fitted parameters for the autoregressive process (CNMF)
-> preprocess.ExtractRawTrace
---
g                           : longblob                      # array with g1, g2, ... values for the AR process
%}


classdef ExtractRawARCoefficients < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end