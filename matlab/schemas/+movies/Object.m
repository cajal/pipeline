%{
# Objects extracted from movies used for generating clips and stills
->stimulus.Movie
---
path                        : varchar(255)                  #
original_file               : varchar(255)                  #
file_template               : varchar(255)                  # filename template with full path
%}

classdef Object < dj.Lookup
    
    methods
        function createClips(obj)
            
            % read data file
            [path,file,file_temp,dur,codec,fps] = fetch1(obj * proj(stimulus.Movie,'codec','file_duration','frame_rate'), ...
                'path', 'original_file', 'file_template', 'file_duration', 'codec','frame_rate');
            
            files = dir(getLocalPath(fullfile(path,file)));
            
            for ifile = 1:length(files)
                if exists(movies.ObjectClip & sprintf('parent_file_name = "%s"',files(ifile).name));continue;end
                infile = fullfile(files(ifile).folder,files(ifile).name);
                try
                    info = ffmpeginfo(infile);
                catch
                    fprintf('%s not supported!',files(ifile).name);
                    continue
                end
                clip_number = floor(info.duration/dur);
                if clip_number<1; continue;end
                if round(info.streams.codec.fps)~=fps; continue; end
                mxclip = max([fetchn(movies.ObjectClip & obj,'clip_number');0]);
                
                % process & insert clips
                for iclip = 1:clip_number
                    tuple = fetch(obj);
                    tuple.clip_number = iclip + mxclip;
                    tuple.file_name = sprintf(file_temp,tuple.clip_number);
                    tuple.parent_file_name = files(ifile).name;
                    if exists(movies.ObjectClip & tuple)
                        continue
                    end
                    
                    % create file
                    start = (iclip-1)*dur;
                    jump_start = max([start - 60,0]);
                    start = start-jump_start;
                    outfile = getLocalPath(fullfile(path,tuple.file_name));
                    if ~exist(outfile,'file')
                        argstr = sprintf('-ss %d -i "%s" -ss %d -t %d %s %s',...
                            jump_start,infile,start,dur,codec,outfile);
                        ffmpegexec(argstr)
                    end
                    
                    % load file & insert
                    fid = fopen(getLocalPath(fullfile(path,tuple.file_name)));
                    tuple.clip = fread(fid,'*int8');
                    fclose(fid);
                    insert(movies.ObjectClip, tuple)
                    delete(outfile)
                end
            end
        end

    end
end
