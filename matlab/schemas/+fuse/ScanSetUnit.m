%{
# individual units corresponding to <module>.ScanSet.Unit
-> `pipeline_experiment`.`scan`
-> `pipeline_shared`.`#pipeline_version`
-> `pipeline_shared`.`#segmentation_method`
unit_id                     : int                           # unique per scan & segmentation method
---
-> fuse.ScanSet
%}


classdef ScanSetUnit < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end