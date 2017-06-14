%{
# Movie clip condition
-> stimulus.Condition
-----
-> stimulus.MovieClip
cut_after              : float           # (s) cuts off after this duration
%}

classdef Clip < dj.Manual & stimulus.core.Visual
    
    properties(Constant)
        version = '1'
        movie_dir = '/home/dimitri/stimuli'
    end
    
    
    methods(Static)
        
        function migrate
            % migrate data from the legacy schema +vis
            
            control = stimulus.getControl;
            
            % migrate monet conditions
            clip = vis.MovieClipCond & (vis.Trial*preprocess.Sync & 'trial_idx between first_trial and last_trial' & 'animal_id>0');
            
            condKeys = clip.fetch';
            count = 0;
            for key = condKeys
                count = count + 1;
                fprintf('[%d/%d]', count, length(condKeys))
                cond = fetch(vis.MovieClipCond & key, '*');
                cond = rmfield(cond, {'animal_id', 'psy_id', 'cond_idx'});
                hash = control.makeConditions(stimulus.Clip, cond);
                
                % copy all trials that used this condition
                for tuple = fetch(vis.Trial*preprocess.Sync & 'trial_idx between first_trial and last_trial' & key,...
                        'last_flip_count->last_flip', 'trial_ts', 'flip_times')'
                    insert(stimulus.Trial, setfield(rmfield(tuple, {'psy_id'}), 'condition_hash', hash{1}))
                end
            end
        end
        
        
        
        function cond = prepare(cond)
            if ~exist(stimulus.Clip.movie_dir, 'dir')
                mkdir(stimulus.Clip.movie_dir)
            end
            
            % get the filename and play the movie if it does not exist
            filename = fetch1(stimulus.MovieClip & cond, 'file_name');
            cond.filename = fullfile(stimulus.Clip.movie_dir, filename);
            if ~exist(cond.filename, 'file')
                fprintf('Writing %s\n', cond.filename)
                fid = fopen(cond.filename, 'w');
                clip = fetch1(stimulus.MovieClip & cond, 'clip');
                fwrite(fid, clip, 'int8');
                fclose(fid);
            end
        end
        
    end
    
    
    methods
        
        function showTrial(self, cond)
            disp(cond.filename)
            [movie, ~, fps] = Screen('OpenMovie', self.win, cond.filename);
            
            screenFPS = round(self.screen.fps);
            frameStep = floor(screenFPS / fps);
            assert(frameStep*fps == screenFPS, 'Screen FPS %d must be an integer multiple of the movie FPS %d', screenFPS, fps);
            
            self.screen.frameStep = frameStep;
            Screen('PlayMovie', movie, 1);
            for i=1:ceil(cond.cut_after*fps)
                tex = Screen('GetMovieImage', self.win, movie);
                if tex<=0
                    break
                end
                Screen('DrawTexture', self.win, tex, [], self.rect)
                self.flip(struct('checkDroppedFrames', i>1))
                Screen('close', tex)
            end
            Screen('CloseMovie', movie)
        end
        
    end
end