%{
# fields that have been processed in the current scan
-> meso.ScanDone
-> meso.Activity
%}


classdef ScanDonePartial < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end