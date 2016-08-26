%{
preprocess.PrepareAodPoint (imported) # points in 3D space in coordinates of an AOD scan
-> preprocess.PrepareAod
point_id        : smallint               # id of a scan point
---
x                           : float                         # (um)
y                           : float                         # (um)
z                           : float                         # (um)
%}


classdef PrepareAodPoint < dj.Relvar
	methods

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end