%{
# grouping table for matched masks
-> preprocess.MatchedScanSite
-> preprocess.Slice
---
translation_correction      : longblob                      # translation correction to account for shifts between scans
%}


classdef MatchedMasks < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end