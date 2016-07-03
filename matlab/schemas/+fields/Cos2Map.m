%{
fields.Cos2Map (computed) # pixelwise cosine fit to directional responses
-> fields.OriMap
---
cos2_amp                    : longblob                      # dF/F at preferred direction
cos2_r2                     : longblob                      # fraction of variance explained (after gaussinization)
cos2_fp                     : longblob                      # p-value of F-test (after gaussinization)
pref_ori                    : longblob                      # (radians) preferred direction
%}


classdef Cos2Map < dj.Relvar & dj.AutoPopulate

	properties
		popRel = fields.OriMap  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end