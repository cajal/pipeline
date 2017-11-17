%{
# scans that are fully processed (updated every time a slice is added)
-> reso.ScanInfo
-> shared.SegmentationMethod
-> shared.SpikeMethod
%}


classdef ScanDone < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end