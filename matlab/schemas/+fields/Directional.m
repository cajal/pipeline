%{
fields.Directional (computed) # all directional drift trials for the scan
-> preprocess.Sync
---
ndirections                 : tinyint                       # number of directions
%}


classdef Directional < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.Sync  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end