%{
# 
-> `pipeline_reso`.`__activity`
-> fuse.Activity
%}


classdef ActivityReso < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end