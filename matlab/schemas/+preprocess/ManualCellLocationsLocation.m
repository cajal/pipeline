%{
# 
-> preprocess.ManualCellLocations
cell_id                     : smallint                      # cell identifier
---
x                           : float                         # x coordinate of the cell
y                           : float                         # y coordinate of the cell
%}


classdef ManualCellLocationsLocation < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end