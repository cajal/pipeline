%{
preprocess.ClusteredEyePos (computed) # my newest table
# add primary key here
-> preprocess.EyeTrackingFrame
-> preprocess.EyePosClusterCenter
-----
%}

classdef ClusteredEyePos < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.EyeTrackingFrame*preprocess.EyePosClusterCenter ;
	end

	methods(Access=protected)

		function makeTuples(self, key)
            
            rv = preprocess.EyePosClusterCenter & struct('animal_id', key.animal_id,...
                                                'session', key.session,...
                                                'scan_idx', key.scan_idx,...
                                                'radius', key.radius,...
                                                'peakorder', key.peakorder) ;
            [cluscenter_x, cluscenter_y] = rv.fetchn('center_x', 'center_y') ;
            rv = preprocess.EyeTrackingFrame & struct('animal_id', key.animal_id,...
                                                'session', key.session,...
                                                'scan_idx', key.scan_idx,...
                                                'frame_id', key.frame_id,...
                                                'eye_quality', key.eye_quality) ;
            eyepos = rv.fetchn('center') ; 
            eyepos = eyepos{1} ;
            if (~isempty(eyepos))
                if (eyepos(1)-cluscenter_x)^2 + (eyepos(2)-cluscenter_y)^2 < key.radius^2
                    self.insert(key) ;
%                    pause(0.001) ;
                end
            end
		end
    end
end