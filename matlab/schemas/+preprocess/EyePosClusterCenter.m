%{
preprocess.EyePosClusterCenter (computed) # my newest table
# add primary key here
-> preprocess.CircleMap
-> preprocess.ClusterPeaks
-----
# add additional attributes
center_x    :   smallint    # center of the cluster in x dimension (camera
                            # pixels)
center_y    :   smallint    # center in y dimension

# add additional attributes
%}

classdef EyePosClusterCenter < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.CircleMap*preprocess.ClusterPeaks ;  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
            rv = preprocess.CircleMap & struct('animal_id', key.animal_id,...
                                                'session', key.session,...
                                                'scan_idx', key.scan_idx,...
                                                'radius', key.radius) ;
            [map, xrange, yrange] = rv.fetchn('map', 'xrange', 'yrange') ;
            map = map{1} ;
            xrange = xrange{1} ;
            yrange = yrange{1} ;
            for ii=1:key.peakorder
                [p,q] = max(map) ;
                [~,s] = max(p) ;
                map = eraseCluster(map,s,q(s),key.radius) ;
            end
            key.center_x = s+xrange(1) ;
            key.center_y = q(s)+yrange(1) ;
			self.insert(key)
		end
    end
end