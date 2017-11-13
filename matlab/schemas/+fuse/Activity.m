%{
# Calcium activity for a single field within a scan
-> `pipeline_experiment`.`scan`
-> `pipeline_shared`.`#pipeline_version`
-> `pipeline_shared`.`#field`
-> `pipeline_shared`.`#channel`
-> `pipeline_shared`.`#segmentation_method`
-> `pipeline_shared`.`#spike_method`
---
-> fuse.Pipe
%}


classdef Activity < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end