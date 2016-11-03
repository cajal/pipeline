%{
preprocess.EyeposCluster (computed) # my newest table
# add primary key here
-> preprocess.EyeTrackingFrame
radius      :   smallint        # radius of the cluster (camera pixels)
center_x    :   smallint        # x coordinate of the cluster's center (camera pixels)
center_y    :   smallint        # y coordinate of cluster's center (camera pixels)
-----
# add additional attributes
%}

classdef EyeposCluster < dj.Relvar & dj.AutoPopulate

	properties
		popRel=preprocess.EyeTrackingFrame;
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end