%{
fields.OriDesignMatrix (computed) # design matrix for directional response
-> fields.Directional
-> fields.CaKernel
---
design_matrix               : longblob                      # times x nConds
regressor_cov               : longblob                      # regressor covariance matrix,  nConds x nConds
%}


classdef OriDesignMatrix < dj.Relvar & dj.AutoPopulate

	properties
		popRel = fields.Directional*fields.CaKernel  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end