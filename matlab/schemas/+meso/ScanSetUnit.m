%{
# single unit in the scan
-> meso.ScanInfo
-> shared.SegmentationMethod
unit_id                     : int                           # unique per scan & segmentation method
---
-> meso.ScanSet
-> meso.FluorescenceTrace
%}


classdef ScanSetUnit < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end