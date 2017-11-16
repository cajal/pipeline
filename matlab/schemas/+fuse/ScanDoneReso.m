%{
# 
-> `pipeline_reso`.`__scan_done`
-> fuse.ScanDone
%}


classdef ScanDoneReso < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end