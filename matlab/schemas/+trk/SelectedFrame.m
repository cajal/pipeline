%{
trk.SelectedFrame (computed) # This schema only contains detected frames that meet a particular quality criterion
-> trk.EyeFrameDetection
-> trk.SelectionProtocol
---
%}


classdef SelectedFrame < dj.Relvar & dj.AutoPopulate

	properties
		popRel = trk.EyeFrameDetection*trk.SelectionProtocol  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end