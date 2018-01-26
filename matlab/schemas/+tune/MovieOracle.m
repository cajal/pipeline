%{
# oracle for repeated videos
-> stimulus.Sync
-> `pipeline_fuse`.`__activity`
%}


classdef MovieOracle < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end