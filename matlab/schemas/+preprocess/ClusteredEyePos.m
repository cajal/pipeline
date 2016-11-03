%{
preprocess.ClusteredEyePos (computed) # my newest table
# add primary key here
-> preprocess.EyeTrackingFrame
-----
center_x    :   smallint    # center of the circle with the largest percentage of points,
                            # x dim, and this point is a member of that
                            # circle
center_y    :   smallint    # center, y dim
radius      :   smallint    # radius of this circle
%}

classdef ClusteredEyePos < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.EyeTrackingFrame ;
	end

	methods(Access=protected)

		function makeTuples(self, key)
            rv = preprocess.EyeTrackingFrame & struct('animal_id', key.animal_id,...
                                                'session', key.session,...
                                                'scan_idx', key.scan_idx,...
                                                'frame_id', key.frame_id,...
                                                'eye_quality', key.eye_quality) ;
            eyepos = rv.fetchn('center') ; 
            eyepos = eyepos{1} ;
            if (~isempty(eyepos))

                rv = preprocess.EyePosClusterCenter & struct('animal_id', key.animal_id,...
                                                'session', key.session,...
                                                'scan_idx', key.scan_idx,...
                                                'peakorder', 1) ; % just look at the clusters around largest peak
                radii = rv.fetchn('radius') ;
                max_points_percent = 0 ;
                max_center_x = 0 ;
                max_center_y = 0 ;
                max_radius = 0 ;
                for ii=1:length(radii)                
                    rv = preprocess.EyePosClusterCenter & struct('animal_id', key.animal_id,...
                                                'session', key.session,...
                                                'scan_idx', key.scan_idx,...
                                                'peakorder', 1,...
                                                'radius', radii(ii)) ; % just look at the clusters around largest peak
                    [clustercenter_x, clustercenter_y, points_percent] = rv.fetchn('center_x', 'center_y', 'points_percent') ;
                    if (eyepos(1)-cluscenter_x)^2 + (eyepos(2)-cluscenter_y)^2 < key.radius^2
                        if (points_percent > max_points_percent)
                            max_points_percent = points_percent ;
                    end
                end
                self.insert(key) ;
            end
		end
    end
end