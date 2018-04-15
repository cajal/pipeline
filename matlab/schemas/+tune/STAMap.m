%{
# receptive field map
-> tune.STA
-> fuse.ActivityTrace
---
map                         : longblob                      # receptive field map
%}


classdef STAMap < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end