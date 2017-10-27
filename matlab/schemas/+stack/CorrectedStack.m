%{
# all slices of each stack after corrections.
-> stack.CorrectionsStitched
%}


classdef CorrectedStack < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end