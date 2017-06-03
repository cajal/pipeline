%{
# pixelwise responses to full-field directional stimuli
-> tuning.OriDesignMatrix
-> `pipeline_preprocess`.`_prepare__galvo_motion`
---
regr_coef_maps              : longblob                      # regression coefficients, widtlh x height x nConds
r2_map                      : longblob                      # pixelwise r-squared after gaussinization
dof_map                     : longblob                      # degrees of in original signal, width x height
%}


classdef OriMapy < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end