%{
preprocess.ExtractRaw (imported) # pre-processing of a twp-photon scan
-> preprocess.Prepare
-> preprocess.Method
---
%}


classdef ExtractRaw < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.Prepare*preprocess.Method & (...
            preprocess.PrepareAod*preprocess.MethodAod | ...
            preprocess.PrepareGalvo*preprocess.MethodGalvo)
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end