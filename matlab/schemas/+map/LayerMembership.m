%{
# 
-> `pipeline_preprocess`.`#slice`
-> `pipeline_preprocess`.`_extract_raw__galvo_r_o_i`
-> `pipeline_preprocess`.`_mask_coordinates`
---
-> experiment.Layer
%}


classdef LayerMembership < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end