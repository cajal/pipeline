%{
# single slice of one stack
-> stack.CorrectedStack
<<<<<<< HEAD
-> `pipeline_shared`.`#channel`
islice                      : smallint                      # index of slice in volume
---
slice                       : longblob                      # image (height x width)
z                           : float                         # slice depth in volume-wise coordinate system
=======
-> shared.Channel
islice                      : smallint                      # index of slice in volume
---
slice                       : longblob                      # image (height x width)
slice_z                     : float                         # slice depth in volume-wise coordinate system
>>>>>>> 0bd758669df75014d62df8c0d09157b457ef65ed
%}


classdef CorrectedStackSlice < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
<<<<<<< HEAD
			 self.insert(key)
=======
% 			 self.insert(key)
>>>>>>> 0bd758669df75014d62df8c0d09157b457ef65ed
		end
	end

end