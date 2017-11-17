%{
# union of ROIs from a stack (usually one per stack)
-> stack.Corrections
volume_id                   : tinyint                       # id of this volume
%}


classdef CorrectionsStitched < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end