%{
# union of ROIs from a stack (usually one volume per stack)
-> stack.Stitching
volume_id                   : tinyint                       # id of this volume
%}


classdef StitchingVolume < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end