%{
# 
-> `pipeline_meso`.`__scan_set`
-> fuse.ScanSet
%}


classdef ScanSetMeso < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end