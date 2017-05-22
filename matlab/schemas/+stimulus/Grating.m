%{
# legacy grating stimulus from the visual_stimulus repo
-> stimulus.Condition
---
monitor_distance_ratio      : float                         # the distance to the screen measured in screen diagonals
pre_blank                   : decimal(4,2)                  # (s) blank period preceding trials
direction                   : decimal(4,1)                  # 0-360 degrees
spatial_freq                : decimal(4,2)                  # cycles/degree
temp_freq                   : decimal(4,2)                  # Hz
luminance                   : float                         # cd/m^2 mean
contrast                    : float                         # Michelson contrast 0-1
aperture_radius             : float                         # in units of half-diagonal, 0=no aperture
aperture_x                  : float                         # aperture x coordinate, in units of half-diagonal, 0 = center
aperture_y                  : float                         # aperture y coordinate, in units of half-diagonal, 0 = center
grating                     : enum('sqr','sin')             # sinusoidal or square, etc.
init_phase                  : float                         # 0..1
trial_duration              : float                         # s, does not include pre_blank duration
phase2_fraction=0           : float                         # fraction of trial spent in phase 2
phase2_temp_freq=0          : float                         # (Hz)
second_photodiode=0         : tinyint                       # 1=paint a photodiode patch in the upper right corner
second_photodiode_time      : decimal(4,1)                  # time delay of the second photodiode relative to the stimulus onset
%}


classdef Grating < dj.Manual & stimulus.core.Visual
    
    properties(Constant)
        version = '1'
    end
    
    
    methods(Static)
        
        function migrate
            % migrate synchronized data from the old schema (+vis/+preprocess)
            % This is not currently debugged since we dont have any data to
            % migrate
            control = stimulus.getControl;
            gratings = (vis.Grating & (preprocess.Sync*vis.Trial & 'trial_idx between first_trial and last_trial')) - proj(stimulus.Trial * stimulus.Grating);
            params = fetch(gratings, '*');
            if isempty(params)
                disp 'nothing to migrate'
            else
                params = rmfield(params, {'animal_id', 'psy_id', 'cond_idx'});
                hashes = control.makeConditions(stimulus.Grating, params);
                error 'Debug when there is data'
            end
        end
        
    end
    
    methods
        
        
        function showTrial(self, cond)
            % execute a single trial with a single cond
            % See PsychToolbox DriftDemo4.m for API calls
            
            % second_photodiode_time should be zero when second_photodiode is zero
            assert(cond.second_photodiode || ~cond.second_photodiode_time)
            assert(cond.second_photodiode_time<cond.trial_duration)
            assert(cond.pre_blank >= -cond.second_photodiode_time)
            % initialized grating
            radius = inf;
            if cond.aperture_radius
                radius = cond.aperture_radius * norm(self.rect(3:4))/2;
            end
            grating = CreateProceduralSineGrating(self.win, self.rect(3), self.rect(4), [0.5 0.5 0.5 0.0], radius);
            
            self.screen.setContrast(cond.luminance, cond.contrast, strcmp(cond.grating,'sqr'))
            phase = cond.init_phase;
            degPerPix = 180/pi/cond.monitor_distance_ratio/norm(self.rect(3:4));
            freq = cond.spatial_freq * degPerPix;  % cycles per pixel
                      
            if cond.pre_blank>0
                if cond.second_photodiode
                    % display black photodiode rectangle during the pre-blank
                    rectSize = [0.05 0.06].*self.rect(3:4);
                    rect = [self.rect(3)-rectSize(1), 0, self.rect(3), rectSize(2)];
                    Screen('FillRect', self.win, 0, rect);
                end
                self.flip(struct('checkDroppedFrames', false))
                WaitSecs(cond.pre_blank + min(0, cond.second_photodiode_time));
                
                if cond.second_photodiode
                    % display black photodiode rectangle during the pre-blank
                    rectSize = [0.05 0.06].*self.rect(3:4);
                    rect = [self.rect(3)-rectSize(1), 0, self.rect(3), rectSize(2)];
                    color = (cond.second_photodiode+1)/2*255;
                    Screen('FillRect', self.win, color, rect);
                    if cond.second_photodiode_time < 0
                        self.flip(struct('logFlips', false))  
                    end
                end
                WaitSecs(max(0, -cond.second_photodiode_time));
            end
            
            % update direction to correspond to 0=north, 90=east, 180=south, 270=west
            direction = cond.direction + 90;
            
            % display drifting grating
            driftFrames1 = floor(cond.trial_duration * (1-cond.phase2_fraction) * self.fps);
            driftFrames2 = floor(cond.trial_duration * cond.phase2_fraction * self.fps);
            phaseIncrement1 = cond.temp_freq/self.fps;
            phaseIncrement2 = cond.phase2_temp_freq/self.fps;
            offset = [cond.aperture_x cond.aperture_y]*norm(self.rect(3:4))/2;
            destRect = self.rect + [offset offset];
            
            % display phase1 grating
            for frame = 1:driftFrames1
                Screen('DrawTexture', self.win, grating, [], destRect, direction, [], [], [], [], ...
                    kPsychUseTextureMatrixForRotation, [phase*360, freq, 0.495, 0]);
                if cond.second_photodiode
                    rectSize = [0.05 0.06].*self.rect(3:4);
                    rect = [self.rect(3)-rectSize(1), 0, self.rect(3), rectSize(2)];
                    if frame/self.fps >= cond.second_photodiode_time
                        color = (cond.second_photodiode+1)/2*255;
                        Screen('FillRect', self.win, color, rect);
                    else
                        Screen('FillRect', self.win, 0, rect);
                    end
                end
                phase = phase + phaseIncrement1;
                self.flip(struct('checkDroppedFrames', frame>1))
            end
            
            % display phase2 grating
            for frame = 1:driftFrames2
                Screen('DrawTexture', self.win, grating, [], destRect, direction, [], [], [], [], ...
                    kPsychUseTextureMatrixForRotation, [phase*360, freq, 0.495, 0]);
                phase = phase + phaseIncrement2;                
                self.flip(struct('checkDroppedFrames', frame>1))
            end
        end
    end
    
end