%{
trk.EyeFrameDetection(computed) # eye frames with detected eye
->trk.EyeFrame
---
pupil_x                     : float                         # pupil x position
pupil_y                     : float                         # pupil y position
pupil_r_minor               : float                         # pupil radius minor axis
pupil_r_major               : float                         # pupil radius major axis
pupil_angle                 : float                         # angle of major axis vs. horizontal axis in radians
pupil_x_std                 : float                         # pupil x position std
pupil_y_std                 : float                         # pupil y position std
pupil_r_minor_std           : float                         # pupil radius minor axis std
pupil_r_major_std           : float                         # pupil radius major axis std
pupil_angle_std             : float                         # angle of major axis vs. horizontal axis in radians
%}
classdef EyeFrameDetection < dj.Relvar & dj.AutoPopulate

    properties
        popRel  = 0;
    end

    methods(Access=protected)

        function makeTuples(self, key)

        end
 end
    
end