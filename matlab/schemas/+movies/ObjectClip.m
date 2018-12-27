%{
# object identities in clips from movies
-> movies.Object
clip_number     : int                                       # clip index
---
file_name                   : varchar(255)                  # full file name
clip                        : longblob                      #
parent_file_name                   : varchar(255)           # parent file name
%}

classdef ObjectClip < dj.Part
    
    properties(SetAccess=protected)
        master= movies.Object
    end
    
    methods
                   function filenames = exportMovie(self, movie_dir)
            % create movie files from clips
            if nargin<2
                movie_dir = 'movies';
            end
            
            if ~exist(movie_dir, 'dir')
                mkdir(movie_dir)
            end
            filenames = {};
            for key = fetch(self)'
                [filename, clip] = fetch1(stimulus.MovieClip & key, 'file_name', 'clip');
                filenames{end+1} = fullfile(movie_dir, filename); %#ok<AGROW>
                if ~exist(filenames{end}, 'file')
                    fprintf('Writing %s\n', filenames{end})
                    fid = fopen(filenames{end}, 'w');
                    fwrite(fid, clip, 'int8');
                    fclose(fid);
                end
            end
        end
        
        function fh = play(self, target_fps, skip_time, cut_after) %create clips
            
            if nargin < 2; target_fps = []; end
            if nargin < 3; skip_time = []; end
            if nargin < 4; cut_after = []; end
            
            % initialize
            set(0,'units','pixels')  ;
            px = get(0,'screensize');
            filename = []; file = []; fps = [];old_filename = [];
            ikey = 0;
            
            % get keys
            keys = fetch(self);
            
            % loop through all clips
            for ikey =1:length(keys)
                
                % load & setup player
                loadNextClip
                f = implay(file, fps);
                f.Parent.Position = [round(px(3)/2-300) round(px(4)/2-200) 600 338];
                uistack(f.Parent)
                if length(keys)==1; f.DataSource.Controls.Repeat = 1; end
                uistack(f.Parent)
                
                % start playing
                play(f.DataSource.Controls)
                drawnow
                f.DataSource.Application.Visual.Axes.Position = [0 0 600 338];
                
                % clear old files
                file = [];
                if exist(old_filename,'file');delete(old_filename);end
                
                % detect if playing
                while  length(keys)>1 && f.isvalid && ~strcmp(f.DataSource.State.CurrentState,'stopped')
                    %loadNextClip % load next clip while waiting -> not
                    %good performance
                    pause(0.1)
                end
                
                % detect if playing else close figure
                if  length(keys)>1
                    if f.isvalid; f.close;else; return; end
                else
                    fh = f;
                end
            end
            
            % delete old files
            if exist(filename,'file')
                delete(filename);
            end
            if exist(old_filename,'file')
                delete(old_filename);
            end
            
            function saveClip(key)
                [filename, clip, frame_rate] = fetch1(movies.ObjectClip * movies.Object * proj(stimulus.Movie,'frame_rate') ...
                    & key, 'file_name', 'clip','frame_rate');
                if isempty(target_fps); fps = frame_rate;else; fps = target_fps; end
                if ~exist(filename, 'file')
                    fid = fopen(filename, 'w');
                    fwrite(fid, clip, 'int8');
                    fclose(fid);
                end
            end
            
            function loadClip
                if isempty(file) % if next file hasn't been loaded yet
                    if ~isempty(skip_time) || ~isempty(cut_after) || (~isempty(target_fps) && target_fps>100)
                        fprintf('Loading next file...%s...',filename)
                        vr = VideoReader(filename);
                        if isempty(skip_time); skip_time = 0;end
                        vr.CurrentTime = skip_time;
                        if isempty(cut_after) || ~cut_after; cut_after = vr.Duration;end
                        frame_rate = vr.FrameRate;
                        skip_frames = ceil(frame_rate\target_fps);
                        file = nan(vr.Height,vr.Width,3,floor(cut_after*frame_rate));
                        for i = 1:cut_after*frame_rate
                            file(:,:,:,i) = vr.readFrame;
                        end
                        file = file(:,:,:,1:skip_frames:end)/255;
                        fps = frame_rate;
                        fprintf('done!\n')
                    else % no file loading
                        file = filename;
                    end
                end
            end
            
            function loadNextClip
                old_filename = filename;
                saveClip(keys(ikey))
                loadClip
            end
        end
             
    end
end

