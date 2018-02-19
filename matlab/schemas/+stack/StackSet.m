%{
# give a unique id to segmented masks in the stack
-> stack.CorrectedStack
-> `pipeline_shared`.`#registration_method`
%}


classdef StackSet < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end