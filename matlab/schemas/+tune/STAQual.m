%{
# 
-> tune.STAMap
---
snr                         : float                         # RF contrast measurement
%}


classdef STAQual < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end