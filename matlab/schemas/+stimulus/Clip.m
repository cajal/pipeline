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
        movie_dir = '/Users/dimitri/stimuli'
    end
    
    
    methods(Static)
        
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
            movie = Screen('OpenMovie', self.win, cond.filename);
            Screen('PlayMovie', movie, 1);
            opts.checkDroppedFrames = false;
            for i=1:ceil(cond.cut_after*self.fps)
                tex = Screen('GetMovieImage', self.win, movie);
                if tex<=0
                    break
                end
                Screen('DrawTexture', self.win, tex, [], self.rect)
                self.flip(opts)
                opts.checkDroppedFrames = true;
                Screen('close', tex)
            end
            Screen('CloseMovie', movie)
        end
        
    end
end