%{
# Trace corresponding to <module>.Activity.Trace
-> fuse.Activity
unit_id                     : int                           # unique per scan & segmentation method
%}


classdef ActivityTrace < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
% 		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end