%{
# single slice of one stack
-> stack.CorrectedStack
-> shared.Channel
islice                      : smallint                      # index of slice in volume
---
slice                       : longblob                      # image (height x width)
slice_z                     : float                         # slice depth in volume-wise coordinate system
%}


classdef CorrectedStackSlice < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end