%{
# cut out of the field in the stack after registration
-> stack.FieldRegistration
---
reg_field                   : longblob                      # 2-d field taken from the stack
%}


classdef FieldRegistrationFieldInStack < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end