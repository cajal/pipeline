%{
preprocess.PrepareGalvoAverageFrame (imported) # average frame for each slice and channel after corrections
-> preprocess.PrepareGalvoMotion
-> preprocess.Channel
---
frame                       : longblob                      # average frame ater Anscombe, max-weighting,
%}


classdef PrepareGalvoAverageFrame < dj.Relvar
    methods
        function saveTiffStack(self, key)
            
            % get movie
            reader = preprocess.getGalvoReader(key);
            movie = reader;
            %movie = reader(:,:,:,:,:);
            
            % get info
            fpf = reader.frames_per_file;
            [nframes, nslices, nchannels,imw,imh] = fetch1(preprocess.PrepareGalvo & key, ...
                'nframes', 'nslices', 'nchannels','px_width','px_height');
            
            % get functions
            fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
            for islice = 1:nslices
                key.slice = islice;
                fixMotion{islice} = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
            end
            
            % initialize
            frameidx = 0; fileidx = 1;
            data = zeros(imw,imh,fpf(fileidx),'uint16');
              
            % make export directory
            [path, fname, ending] = fileparts(getLocalPath(reader.files{fileidx}));
            if ~isdir(fullfile(getLocalPath(path),'mrc'))
                mkdir(getLocalPath(fullfile(path,'mrc')))
            end
            
            % set tiff params
            options.append = true;
                  
            % save
            tic
            for iframe = 1:nframes
                for islice = 1:nslices
                    for ichannel = 1:nchannels
                        frameidx = frameidx+1;
                        data(:,:,frameidx) = uint16(fixMotion{islice}(fixRaster(movie(:,:,ichannel, islice, iframe)), iframe));
                                               
                        if frameidx == fpf(fileidx)
                            [path, fname, ending] = fileparts(getLocalPath(reader.files{fileidx}));
                            name = fullfile(path,'mrc',sprintf('%s%s',fname,ending));  
                            fprintf('Writing file %s', name);
                            pipetools.saveastiff(data,name)
                            data = zeros(imw,imh,fpf(fileidx),'uint16');
                            fileidx = fileidx+1;
                            frameidx = 0;
                        end
                    end
                end
                
                if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                    fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                end
            end
        end
        
        function save(self)
            for key = fetch(preprocess.PrepareGalvoMotion & self)'
                frames = fetchn(preprocess.PrepareGalvoAverageFrame & key, 'frame', 'ORDER BY channel DESC');
                path = fullfile(pwd, sprintf('frame%05u-%05u-%u.png', ...
                    key.animal_id, key.scan_idx, key.slice));
                fprintf('saving %s...', path)
                frames = cellfun(@(f) sqrt((f-min(f(:)))/(max(f(:))-min(f(:)))+0.01)-0.1, frames, 'uni', false);
                frames = cat(3,frames{:});
                if size(frames,3) == 2
                    frames(:,:,3)=0;
                end
                imwrite(frames, path)
                disp done.
            end
        end
        
        function makeTuples(self, key, movie)
            q = 6;
            [nframes, nslices, nchannels] = fetch1(preprocess.PrepareGalvo & key, ...
                'nframes', 'nslices', 'nchannels');
            % average frame
            fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
            for islice = 1:nslices
                key.slice = islice;
                fixMotion = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
                for ichannel = 1:nchannels
                    tic
                    tuple = key;
                    tuple.channel = ichannel;
                    fprintf('Averaging slice %d/%d channel %d\n', islice, nslices, ichannel)
                    frame = 0;
                    for iframe = 1:nframes
                        frame = frame + max(0,fixMotion(fixRaster(movie(:,:,ichannel, islice, iframe)), iframe)).^q;
                        if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                            fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                        end
                    end
                    tuple.frame = single((frame/nframes).^(1/q));
                    self.insert(tuple)
                end
            end
        end
    end
end
