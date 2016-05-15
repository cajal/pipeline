%{
pre.AlignMotion (imported) # motion correction
-> pre.AlignRaster
-> pre.Slice
---
-> pre.Channel
motion_xy                   : longblob                      # (pixels) y,x motion correction offsets
motion_rms                  : float                         # (um) stdev of motion
align_times=CURRENT_TIMESTAMP: timestamp                    # automatic
INDEX(animal_id,session,scan_idx,channel)
%}

classdef AlignMotion < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.AlignRaster
    end
    
    
    methods
        function fun = get_fix_motion_fun(self)
            xy = self.fetch1('motion_xy');
            fun = @(frame, i) ne7.ip.correctMotion(frame, xy(:,i));
        end
        
        function saveMovie(self, start_time, duration, speedup, fps)
            keys = self.fetch;
            assert(count(pre.ScanInfo & keys) == 1, 'One scan at a time please')
            fixRaster = get_fix_raster_fun(pre.AlignRaster & keys);
            fixMotion = arrayfun(@(key) get_fix_motion_fun(pre.AlignMotion & key), keys, 'uni', false);
            [nframes, caFps] = fetch1(pre.ScanInfo & keys, 'nframes', 'fps');
            reader = pre.getReader(keys(1));
            frames = max(1, min(nframes, round(start_time*caFps + (1:duration*caFps))));
            slices = [keys.slice];
            movie = double(reader(:,:,:,slices,frames));
            for i = 1:length(frames)
                for ichannel = 1:reader.nchannels
                    for islice = 1:length(slices)
                        movie(:,:,ichannel,islice,i) = ...
                            fixMotion{islice}(fixRaster(movie(:,:,ichannel,islice,i)), frames(i));
                    end
                end
            end
            movie = sqrt(max(0,movie));
            for islice = 1:length(slices)
                for ichannel = 1:reader.nchannels
                    q = movie(:,:,ichannel,islice,:);
                    q = min(1,q/quantile(q(:),0.999));
                    sz = size(q);
                    q = reshape(q,[],sz(end));
                    q = resample(q', round(fps), round(caFps*speedup))';
                    q = reshape(q,sz(1),sz(2),1,[]);
                    if ichannel==1
                        m = q;
                    else
                        m = cat(3,q,m);
                    end
                end
                m(:,:,3,:)=0;   % make 3D
                m = max(0, min(1, m));
                if islice==1
                    r = m;
                else
                    c = 1 - 0.5*(islice - 1)/length(slices);
                    for iframe=1:size(m,4)
                        m(:,:,:,iframe) = hsv2rgb(bsxfun(@times, rgb2hsv(m(:,:,:,iframe)), cat(3, 1, c, c)));
                    end
                    r = r + m;
                end
            end
            r = r/max(r(:));
            
            filename = sprintf('~/Desktop/camovie%05d-%03d', keys(1).animal_id, keys(1).scan_idx);
            v = VideoWriter(filename, 'MPEG-4');
            v.FrameRate = fps;
            v.Quality = 100;
            open(v)
            writeVideo(v, uint8(r*255));
            close(v)            
        end
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            tic
            [quantal_size, max_intensity, keys] = fetchn(pre.ScanCheck & key, ...
                'quantal_size', 'max_intensity');
            key = keys(argmax(max_intensity./quantal_size));
            [zero, quantal_size] = fetch1(pre.ScanCheck & key, ...
                'min_var_intensity', 'quantal_size');
            
            anscombe = @(img) 2*sqrt(max(0, img-zero)/quantal_size+3/8);   % Anscombe transform
            
            reader = pre.getReader(key);
            fixRaster = get_fix_raster_fun(pre.AlignRaster & key);
            getFrame = @(islice, iframe) fixRaster(anscombe(double(reader(:,:,key.channel,islice,iframe))));
            sz = size(reader);
            
            k = gausswin(41); k=k/sum(k);
            sharpen = @(im) im-imfilter(imfilter(im,k,'symmetric'),k','symmetric');
            taper = 40;  % the larger the number the thinner the taper
            mask = atan(taper*hanning(sz(1)))*atan(taper*hanning(sz(2)))'/atan(taper)^2;
            templateStack = fetch1(pre.ScanCheck & key, 'template');
            nslices = fetch1(pre.ScanInfo & key, 'nslices');
            assert(size(templateStack,3) == nslices)
            for islice = 1:nslices
                key.slice = islice;
                template = templateStack(:,:,islice);
                template = sharpen(mask.*(template - mean(template(:))));
                
                ftemplate = conj(fft2(double(template)));
                nframes = reader.nframes;
                xy = nan(2,nframes);
                avgFrame = 0;
                
                for iframe = 1:nframes
                    if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                        fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                    end
                    frame = getFrame(islice, iframe);
                    [x, y] = ne7.ip.measureShift(fft2(frame).*ftemplate);
                    xy(:,iframe) = [x;y];
                    avgFrame = avgFrame + ne7.ip.correctMotion(frame, [x;y])/nframes;
                end
                key.motion_xy = xy;
                
                [dx,dy] = fetch1(pre.ScanInfo & key, 'um_width/px_width->dx', 'um_height/px_height->dy');
                xy = bsxfun(@times, [dx; dy], xy);  % convert to microns
                xy = bsxfun(@minus, xy, mean(xy,2));  % subtract mean
                d = sqrt(sum(xy.^2));   % distance from average position
                key.motion_rms = sqrt(mean(d.^2));   % root mean squared distance
                
                self.insert(key)
            end
        end
    end
    
end



function j = argmax(r)
[~,j] = max(r);
end
