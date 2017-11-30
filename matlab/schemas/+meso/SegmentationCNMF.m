%{
# source extraction using constrained non-negative matrix factorization
-> meso.Segmentation
---
params                      : varchar(1024)                 # parameters send to CNMF as JSON array
%}


classdef SegmentationCNMF < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end