%{
# classification of segmented masks.
-> meso.Segmentation
-> `pipeline_shared`.`#classification_method`
---
classif_time=CURRENT_TIMESTAMP: timestamp                   # automatic
%}


classdef MaskClassification < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end