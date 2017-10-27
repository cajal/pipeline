%{
# stack corrections
-> stack.CorrectionChannel
-> stack.StackInfo
%}


classdef Corrections < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end