%{
tuning.MonetCleanRF (computed) # RF maps with common components removed
-> tuning.MonetRFMap
---
clean_map                   : longblob                      # 
%}


classdef MonetCleanRF < dj.Relvar & dj.AutoPopulate

	properties
		popRel = tuning.MonetRFMap  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end