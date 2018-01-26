%{
# different quality metrics for a scan (before corrections)
-> meso.ScanInfo
%}


classdef Quality < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end