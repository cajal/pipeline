%{
# 
-> reso.MaskClassification
-> reso.SegmentationMask
---
-> shared.MaskType
%}


classdef MaskClassificationType < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end