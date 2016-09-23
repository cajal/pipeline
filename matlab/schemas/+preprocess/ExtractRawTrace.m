%{
preprocess.ExtractRawTrace (imported) # raw trace, common to Galvo
-> preprocess.ExtractRaw
-> preprocess.Channel
trace_id        : smallint               # 
---
raw_trace                   : longblob                      # unprocessed calcium trace
%}


classdef ExtractRawTrace < dj.Relvar
	methods

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			self.insert(key)
		end
	end

end