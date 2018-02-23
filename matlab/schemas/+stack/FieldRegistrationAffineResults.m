%{
# some intermediate results from affine registration
-> stack.FieldRegistration
---
score_map                   : longblob                      # 3-d map of best correlation scores for each yaw, pitch, rol combination
position_map                : longblob                      # 3-d map of best positions (x, y, z) for each yaw, pitch, roll combination
%}


classdef FieldRegistrationAffineResults < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end