%{
# Relative scan coordinates to a reference map
-> experiment.Scan
slice                       : tinyint                       # slice id
---
x_offset                    : double                        # x center coordinate in pixels
y_offset                    : double                        # y center coordinate in pixels
depth                       : double                        # depth of slice from surface in microns
tform                       : mediumblob                    # transformation matrix for rotation,scale,flip relative to vessel map
pxpitch                     : double                        # estimated pixel pitch of the reference map (px)
ref_key                     : mediumblob                    # key of the reference vessel map
%}


classdef ScanCoordinates < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end