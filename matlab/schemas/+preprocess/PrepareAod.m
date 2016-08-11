%{
preprocess.PrepareAod (imported) # information about AOD scans
-> preprocess.Prepare
---
%}


classdef PrepareAod < dj.Relvar
	methods

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end