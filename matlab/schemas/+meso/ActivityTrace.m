%{
# deconvolved calcium acitivity
-> meso.ScanSetUnit
-> `pipeline_shared`.`#spike_method`
---
-> meso.Activity
trace                       : longblob                      # 
%}


classdef ActivityTrace < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end