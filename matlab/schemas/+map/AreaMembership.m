%{
# brain area membership of cells
-> `pipeline_preprocess`.`_extract_raw__galvo_r_o_i`
---
-> experiment.BrainArea
%}


classdef AreaMembership < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end