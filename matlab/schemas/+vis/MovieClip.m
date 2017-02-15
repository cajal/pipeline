%{
vis.MovieClip (lookup) # clips from movies
-> vis.Movie
clip_number     : int                    # clip index
---
file_name                   : varchar(255)                  # full file name
clip                        : longblob                      #
%}


classdef MovieClip < dj.Relvar
    
    methods
        
        function filenames = export(self,movie_dir) %create clips
            if nargin<2
               movie_dir = 'movies';
            end
            
            if ~exist(movie_dir, 'dir')
                mkdir(movie_dir)
            end
            filenames = {};
            for key = fetch(self)'
                [filename, clip] = fetch1(vis.MovieClip & key, 'file_name', 'clip');
                filenames{end+1} = fullfile(movie_dir, filename); %#ok<AGROW>
                if ~exist(filenames{end}, 'file')
                    fprintf('Writing %s\n', filenames{end})
                    fid = fopen(filenames{end}, 'w');
                    fwrite(fid, clip, 'int8');
                    fclose(fid);
                end
            end
        end
    end
end