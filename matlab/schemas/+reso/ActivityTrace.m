%{
# deconvolved calcium acitivity
-> reso.ScanSetUnit
-> `pipeline_shared`.`#spike_method`
---
-> reso.Activity
trace                       : longblob                      # 
%}


classdef ActivityTrace < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end