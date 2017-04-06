%{
vis.Movie (lookup) # movies used for generating clips and stills
movie_name      : char(8)                # short movie title
---
path                        : varchar(255)                  # 
movie_class                 : enum('mousecam','object3d','madmax') # 
original_file               : varchar(255)                  # 
file_template               : varchar(255)                  # filename template with full path
file_duration               : float                         # (s) duration of each file (must be equal)
codec="-c:v libx264 -preset slow -crf 5": varchar(255)      # 
movie_description           : varchar(255)                  # full movie title
frame_rate=30               : float                         # frames per second
frame_width=256             : int                           # pixels
frame_height=144            : int                           # pixels
params=null                 : longblob                      # movie parameters for parametric models
%}


classdef Movie < dj.Relvar
    
    methods 
        function createClips(obj)
            [path,file,file_temp,dur,codec] = fetch1(obj,'path','original_file','file_template','file_duration','codec');
            
            infile = getLocalPath(fullfile(path,file));
            info = ffmpeginfo(infile);
            clip_number = floor(info.duration/dur);
            
            % read data file
            csvname = [infile(1:end-3) 'csv'];
            if exist(csvname,'file')
                data = csvread(csvname,1,0); %#ok<NASGU>
                fileID = fopen(csvname,'r');
                names = textscan(fileID, '%s', 1, 'delimiter', '\n', 'headerlines', 0);
                fclose(fileID);
                names = textscan(names{1}{1},'%s','delimiter',',');
                names = names{1};
                params = [];
                for iname = 1:length(names)
                    eval(['params.' names{iname} '=data(:,iname);']);
                end        
            end
            
            % update movie params
            update(obj, 'params', params)
            update(obj, 'frame_rate', info.streams.codec.fps)
            
            % process & insert clips
            for iclip = 1:clip_number
                tuple = fetch(obj);
                tuple.clip_number = iclip;
                tuple.file_name = sprintf(file_temp,iclip);
                if exists(vis.MovieClip & tuple)
                    continue
                end
               
                % create file
                start = (iclip-1)*dur;
                outfile = getLocalPath(fullfile(path,tuple.file_name));
                if ~exist(outfile,'file')
                    argstr = sprintf('-i %s -ss %d -t %d %s %s',infile,start,dur,codec,outfile);
                    ffmpegexec(argstr)
                end
                
                % load file & insert
                fid = fopen(getLocalPath(fullfile(path,tuple.file_name)));
                tuple.clip = fread(fid,'*int8');
                fclose(fid);
                insert(vis.MovieClip,tuple)
                delete(outfile)
            end
        end
    end
end
