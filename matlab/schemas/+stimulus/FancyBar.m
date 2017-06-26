%{
# Moving Bar stimulus that keeps the size and speed of the bar constant relative to the mouse�s perspective.
-> stimulus.Condition
---
monitor_distance            : decimal(4,1)                  # (cm) eye-to-monitor distance
monitor_size                : decimal(5,2)                  # (inches) size diagonal dimension
monitor_aspect              : decimal(4,3)                  # physical aspect ratio of monitor
resolution_x                : smallint                      # (pixels) display resolution along x
resolution_y                : smallint                      # (pixels) display resolution along y
fps                         : decimal(5,2)                  # display refresh rate
pre_blank                   : double                        # (s) blank period preceding trials
luminance                   : float                         # (cd/m^2)
contrast                    : float                         # michelson contrast
bar_width                   : float                         # Degrees
grid_width                  : float                         # Degrees
bar_speed                   : float                         # Bar speed in deg/s
flash_speed                 : float                         # cycles/sec temporal frequency of the grid flickering
style                       : enum('grating','checkerboard')# selection beween a bar with a flashing checkeboard or a moving grating
grat_width                  : float                         # in cycles/deg
grat_freq                   : float                         # in cycles/sec
axis                        : enum('vertical', 'horizontal')# the direction of bar movement

%}

classdef FancyBar < dj.Manual & stimulus.core.Visual
    
    properties(Constant)
        version = '1'   
    end
    
    methods(Static)
        function migrate
            % migrate synchronized trials from +vis, incrementally
            control = stimulus.getControl;
            control.clearAll
            scans = experiment.Scan & (preprocess.Sync*vis.Trial*vis.FancyBar & 'trial_idx between first_trial and last_trial');
            remain = scans - stimulus.Trial * stimulus.FancyBar; 
            
            for scanKey = remain.fetch'
                disp(scanKey)
                geometry = rmfield(fetch(experiment.DisplayGeometry & scanKey, '*'), {'display_timestamp', 'session'});
                assert(length(geometry)==1, 'DisplayGeometry is missing')
                
                params = fetch(vis.FancyBar & (vis.Trial * preprocess.Sync & scanKey & 'trial_idx between first_trial and last_trial'), '*');
                params = dj.struct.join(params, geometry);
                hashes = control.makeConditions(stimulus.FancyBar, rmfield(params, {'animal_id', 'psy_id', 'cond_idx'}));
                trials =  fetch(vis.Trial & vis.FancyBar & (preprocess.Sync & scanKey & 'trial_idx between first_trial and last_trial'), '*', 'last_flip_count->last_flip');
                hashes = hashes(arrayfun(@(trial) find([params.cond_idx]==trial.cond_idx, 1, 'first'), trials));
                trials = rmfield(trials, {'psy_id', 'cond_idx'});
                [trials.condition_hash] = deal(hashes{:});
                insert(stimulus.Trial, dj.struct.join(trials, scanKey))
            end
        end
    end
    
    
    methods

        function showTrial(self, cond)
            % execute a single trial with a single cond
            % See PsychToolbox DriftDemo4.m for API calls
            
            assert(all(ismember({
                'pre_blank'
                'luminance'
                'contrast'
                'bar_width'
                'grid_width'
                'bar_speed'
                'flash_speed'
                'style'
                'grat_width'
                'grat_freq'
                'axis'
                }, fieldnames(cond))))
            
            self.screen.setContrast(cond.luminance, cond.contrast)
            self.flip(struct('logFlips', false, 'checkDroppedFrames', false))
            WaitSecs(cond.pre_blank);
            
            % stimulus resolution
            reduceFactor = 7; % for cpu/ram compatibility. default:7
            
            % some member variables..
            x0 = cond.monitor_distance;
            
            % CONVERT TO SCREEN UNITS
            % setup parameters
            ymonsize =  cond.monitor_size*2.54/sqrt(1+cond.monitor_aspect^2);% cm Y monitor size
            xmonsize =ymonsize*cond.monitor_aspect;% cm X monitor size
            monSize = [xmonsize ymonsize];
            
            % calculatestaff
            if strcmp(cond.axis,'vertical'); axis = 2; else axis=1;end
            FoV = atand(monSize(axis)/2/x0)*2;
            GridCyclesPerRadiant = 180/cond.grid_width/pi/2; % spatial frequency of the grid in cycles per radiant
            BarCyclesPerRadiant = 180/cond.bar_width/pi/2; % spatial frequency of the bar in cycles per radiant
            BarCyclesPerSecond = cond.bar_speed/FoV; % convert to cycles per second.
            BarOffsetCyclesPerFrame = BarCyclesPerSecond / self.screen.fps;
            GridOffsetCyclesPerFrame = cond.flash_speed / self.screen.fps;
            GratOffsetCyclesPerFrame = cond.grat_freq/self.screen.fps;
            
            % initialize vectors
            y = linspace(-(ymonsize/2),ymonsize/2,cond.resolution_y/reduceFactor);
            z = linspace(-(xmonsize/2),xmonsize/2,cond.resolution_x/reduceFactor);
            [Y,Z] = ndgrid(y,z);
            
            % create tranformations of space
            theta = pi/2-acos(Y./sqrt(x0^2+Y.^2+Z.^2)); % vertical
            phi = atan(-Z/x0); % horizontal
            
            % create grid
            VG1 = (cos(2*pi*GridCyclesPerRadiant*(theta)))>0; % vertical grading
            VG2 = (cos((2*pi*GridCyclesPerRadiant*(theta))-pi))>0; % vertical grading with pi offset
            HG = cos(2*pi*GridCyclesPerRadiant*phi)>0; % horizontal grading
            Grid = bsxfun(@times,VG1,HG) + bsxfun(@times,VG2,1-HG); %combine all
            
            % How many frames do we have to precalculate?
            delay = (atand(monSize(axis)/2/x0)*2 + cond.bar_width)/cond.bar_speed*1000 ; % in miliseconds
            nbFrames = ceil(delay/1000*self.screen.fps);
            
            % angle
            if axis==1;angle1 = phi; angle2 = theta;else angle1 = theta;angle2 = phi;end
            
            % intialize offsets
            BarOffset = 0;
            GridOffset = 0;
            GratOffset = 0;
            barHalfWidth = pi/2;
            BarOffsetCyclesPerFrame = BarOffsetCyclesPerFrame*FoV/cond.bar_width/2;
            startOffset = min(angle1(:)*BarCyclesPerRadiant) - 1/4;
            screenRect = Screen('Rect', self.win);
            
            % randomize stimulus phase %%%% fix this %%%%
            op =  sign(randn);
            
            for i=1:nbFrames
                
                angle = 2*pi*(angle1*BarCyclesPerRadiant+BarOffset+startOffset);
                angle(angle<(-barHalfWidth) | angle>(barHalfWidth)) = pi; % threshold grading to create a single bar
                A = cos(angle)>0; % squaring
                
                switch cond.style
                    case 'grating'
                        gred = (cos(2*pi*(cond.grat_width*(angle2)+GratOffset*op)))>0; % vertical grading
                        texMat = uint8(A.*abs(gred)*254);
                    case 'checkerboard'
                        flash = cos(2*pi*GridOffset)>0;
                        texMat =  uint8(A.*abs(Grid-flash)*254);
                end
                
                tex = Screen('MakeTexture',self.win,texMat);
                Screen('DrawTexture',self.win,tex,[],screenRect);
                
                self.screen.flip(struct('checkDroppedFrames', i>=1))
                
                GratOffset = GratOffset+ GratOffsetCyclesPerFrame;
                GridOffset = GridOffset+ GridOffsetCyclesPerFrame;
                BarOffset = BarOffset+BarOffsetCyclesPerFrame;
                Screen('Close',tex);                
            end
        end
        
    end
    
end