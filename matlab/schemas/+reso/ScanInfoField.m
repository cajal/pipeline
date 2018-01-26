%{
# slice-specific scan information
-> reso.ScanInfo
-> `pipeline_shared`.`#field`
---
z                           : float                         # (um) absolute depth with respect to the surface of the cortex
delay_image                 : longblob                      # 
%}


classdef ScanInfoField < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end