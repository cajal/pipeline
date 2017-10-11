%{
# mask center of mass of a segmented cell
-> map.ScanCoordinates
-> `pipeline_preprocess`.`_extract_raw__galvo_r_o_i`
---
xloc                        : double                        # x location in microns relative to the center of reference map
yloc                        : double                        # y location in microns relative to the center of reference map
zloc                        : double                        # z location in micro meters relative to the surface
%}


classdef MaskCoordinates < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end