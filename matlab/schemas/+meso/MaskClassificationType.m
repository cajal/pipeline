%{
# 
-> meso.MaskClassification
-> meso.SegmentationMask
---
-> `pipeline_shared`.`#mask_type`
%}


classdef MaskClassificationType < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end