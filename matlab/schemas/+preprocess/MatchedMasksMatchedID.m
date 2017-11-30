%{
# 
-> preprocess.MatchedMasks
-> preprocess.ExtractRawGalvoROI
other_trace_id              : smallint                      # index of the other scan
%}


classdef MatchedMasksMatchedID < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end