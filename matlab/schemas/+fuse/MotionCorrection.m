%{
# calcium activity for the whole scan (multiple scan fields)
-> experiment.Scan
-> `pipeline_shared`.`#pipeline_version`
-> `pipeline_shared`.`#field`
---
-> fuse.Pipe
%}


classdef MotionCorrection < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end