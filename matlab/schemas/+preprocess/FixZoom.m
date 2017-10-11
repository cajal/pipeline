%{
# table for fixing faulty calculation in preprocess.PrepareGalvo
-> preprocess.PrepareGalvo
---
um_width                    : float                         # width in microns
um_height                   : float                         # height in microns
%}


classdef FixZoom < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end