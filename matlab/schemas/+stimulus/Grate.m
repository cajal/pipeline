%{
# simple grating with aperture
-> stimulus.Condition
---
monitor_distance_ratio      : float                         # the distance to the screen measured in screen diagonals
pre_blank=0                 : decimal(4,2)                  # (s) blank period preceding the trial
direction                   : decimal(4,1)                  # 0-360 degrees
spatial_freq                : decimal(4,2)                  # cycles/degree
temp_freq                   : decimal(4,2)                  # Hz
aperture_radius=0           : float                         # in units of half-diagonal, 0=no aperture
aperture_x=0                : float                         # aperture x coordinate, in units of half-diagonal, 0 = center
aperture_y=0                : float                         # aperture y coordinate, in units of half-diagonal, 0 = center
init_phase                  : float                         # 0..1
duration                    : float                         # s, does not include pre_blank duration
%}


classdef Grate < dj.Manual & stimulus.core.Visual
    
    properties(Constant)
        version = '1'
    end
    
    
    methods
        
        function showTrial(self, cond)
            % execute a single trial with a single cond
            % See PsychToolbox DriftDemo4.m for API calls
            radius = inf;
            if cond.aperture_radius
                radius = cond.aperture_radius * norm(self.rect(3:4))/2;
            end
            grating = CreateProceduralSineGrating(self.win, self.rect(3), self.rect(4), [0.5 0.5 0.5 0.0], radius);
            
            % square gratings are achived by setting the contrast very high in setContrast
            phase = cond.init_phase;
            degPerPix = 180/pi/cond.monitor_distance_ratio/norm(self.rect(3:4));
            freq = cond.spatial_freq * degPerPix;  % cycles per pixel
            
            if cond.pre_blank>0
                self.flip(struct('checkDroppedFrames', false))  %  the extra flip is logged with pre-blank
                WaitSecs(cond.pre_blank);
            end
            
            % update direction to correspond to 0=north, 90=east, 180=south, 270=west
            direction = cond.direction + 90;
            
            % display drifting grating
            driftFrames = floor(cond.duration * self.fps);
            phaseStep = cond.temp_freq/self.fps;
            offset = [cond.aperture_x cond.aperture_y]*norm(self.rect(3:4))/2;
            destRect = self.rect + [offset offset];
            
            % display phase1 grating
            for frame = 1:driftFrames
                Screen('DrawTexture', self.win, grating, [], destRect, direction, [], [], [], [], ...
                    kPsychUseTextureMatrixForRotation, [phase*360, freq, 0.495, 0]);
                phase = phase + phaseStep;
                self.flip(struct('checkDroppedFrames', frame>1))
            end
            
        end
    end
    
end