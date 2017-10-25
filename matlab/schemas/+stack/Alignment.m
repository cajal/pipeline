%{
# inter-slice alignment
-> stack.MotionCorrection
---
y_shifts                    : longblob                      # 
x_shifts                    : longblob                      # 
%}


classdef Alignment < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end