%{
fields.OriMap (imported) # pixelwise responses to full-field directional stimuli
-> fields.OriDesignMatrix
-> preprocess.PrepareGalvoMotion
---
regr_coef_maps              : longblob                      # regression coefficients, width x height x nConds
r2_map                      : longblob                      # pixelwise r-squared after gaussinization
dof_map                     : longblob                      # degrees of in original signal, width x height
%}


classdef OriMap < dj.Relvar & dj.AutoPopulate

	properties
		popRel = fields.OriDesignMatrix*preprocess.PrepareGalvoMotion  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end