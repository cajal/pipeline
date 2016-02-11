%{
trk.EyeFrame (computed) # eye tracking info for each frame of a movie
-> rf.Eye
-> trk.SVM
frame           : int                    # frame number in movie
---
eye_frame_ts=CURRENT_TIMESTAMP: timestamp                   # automatic
%}


classdef EyeFrame < dj.Relvar & dj.AutoPopulate

	properties
		popRel = rf.Eye*trk.SVM  
	end

	methods(Access=protected)

		function makeTuples(self, key)
            error 'This table is populated from python'
		end
	end

end