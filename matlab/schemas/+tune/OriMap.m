%{
# Pixelwise orientation response map
-> tune.OriDesign
-> tune.CaMovie
---
response_map                : longblob                      # pixelwise normalized response
activity_map                : longblob                      # root of sum of squares
%}


classdef OriMap < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end