%{
# Inferred background components with the CNMF algorithm
-> preprocess.ExtractRaw
-> preprocess.Channel
-> preprocess.Slice
---
masks                       : longblob                      # array (im_width x im_height x num_background_components)
activity                    : longblob                      # array (num_background_components x timesteps)
%}


classdef ExtractRawBackgroundComponents < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end