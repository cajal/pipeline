%{
# Each pixel shows the (average) temporal correlation between that pixel and its four neighbors
-> preprocess.ExtractRaw
-> preprocess.Channel
-> preprocess.Slice
---
correlation_image           : longblob                      # correlation image
%}


classdef ExtractRawGalvoCorrelationImage < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end