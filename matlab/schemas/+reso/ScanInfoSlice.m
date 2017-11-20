%{
# slice-specific scan information
-> reso.ScanInfo
-> shared.Slice
---
z                           : float                         # (um) absolute depth with respect to the surface of the cortex
%}


classdef ScanInfoSlice < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end