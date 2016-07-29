%{
preprocess.PrepareGalvoMotion (imported) # motion correction for galvo scans
-> preprocess.PrepareGalvo
-> preprocess.Slice
---
-> preprocess.Channel
template                    : longblob                      # stack that was used as alignment template
motion_xy                   : longblob                      # (pixels) y,x motion correction offsets
motion_rms                  : float                         # (um) stdev of motion
align_times=CURRENT_TIMESTAMP: timestamp                    # automatic
%}


classdef PrepareGalvoMotion < dj.Relvar
    methods
        
        function fun = get_fix_motion_fun(self)
            xy = self.fetch1('motion_xy');
            fun = @(frame, i) ne7.ip.correctMotion(frame, xy(:,i));
        end
        
%         function saveMovie(self, start_time, duration, speedup, fps)
%             error 'update for preprocess schema'
%             fixRaster = get_fix_raster_fun(pre.AlignRaster & keys);
%             fixMotion = arrayfun(@(key) get_fix_motion_fun(pre.AlignMotion & key), keys, 'uni', false);
%             [nframes, caFps] = fetch1(pre.ScanInfo & keys, 'nframes', 'fps');
%             reader = pre.getReader(keys(1));
%             frames = max(1, min(nframes, round(start_time*caFps + (1:duration*caFps))));
%             slices = [keys.slice];
%             movie = double(reader(:,:,:,slices,frames));
%             for i = 1:length(frames)
%                 for ichannel = 1:reader.nchannels
%                     for islice = 1:length(slices)
%                         movie(:,:,ichannel,islice,i) = ...
%                             fixMotion{islice}(fixRaster(movie(:,:,ichannel,islice,i)), frames(i));
%                     end
%                 end
%             end
%             movie = sqrt(max(0,movie));
%             for islice = 1:length(slices)
%                 for ichannel = 1:reader.nchannels
%                     q = movie(:,:,ichannel,islice,:);
%                     q = min(1,q/quantile(q(:),0.999));
%                     sz = size(q);
%                     q = reshape(q,[],sz(end));
%                     q = resample(q', round(fps), round(caFps*speedup))';
%                     q = reshape(q,sz(1),sz(2),1,[]);
%                     if ichannel==1
%                         m = q;
%                     else
%                         m = cat(3,q,m);
%                     end
%                 end
%                 m(:,:,3,:)=0;   % make 3D
%                 m = max(0, min(1, m));
%                 if islice==1
%                     r = m;
%                 else
%                     c = 1 - 0.5*(islice - 1)/length(slices);
%                     for iframe=1:size(m,4)
%                         m(:,:,:,iframe) = hsv2rgb(bsxfun(@times, rgb2hsv(m(:,:,:,iframe)), cat(3, 1, c, c)));
%                     end
%                     r = r + m;
%                 end
%             end
%             r = r/max(r(:));
%             
%             filename = sprintf('~/Desktop/camovie%05d-%03d', keys(1).animal_id, keys(1).scan_idx);
%             v = VideoWriter(filename, 'MPEG-4');
%             v.FrameRate = fps;
%             v.Quality = 100;
%             open(v)
%             writeVideo(v, uint8(r*255));
%             close(v)
%         end
%     end
    
        
        function makeTuples(self, key, reader, movie)
            tic
            fprintf('Motion alignment: ')
            % prepare
            [nframes, nslices] = fetch1(preprocess.PrepareGalvo & key, ...
                'nframes', 'nslices');
            [dx,dy] = fetch1(preprocess.PrepareGalvo & key, ...
                'um_width/px_width->dx', 'um_height/px_height->dy');
            zero = 0;
            switch fetch1(experiment.SessionFluorophore & key, 'fluorophore')
                case 'RCaMP1a'
                    key.channel = 2;
                otherwise
                    key.channel = 1;
            end
            quantal_size = 50;  % from prior experience
            anscombe = @(img) 2*sqrt(max(0, img-zero)/quantal_size+3/8);   % Anscombe transform
            fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
            getFrames = @(islice, iframe) fixRaster(anscombe(double(movie(:,:,key.channel,islice,iframe))));
            sz = size(movie);
            k = gausswin(41); k=k/sum(k);
            sharpen = @(im) im-imfilter(imfilter(im,k,'symmetric'),k','symmetric');
            smoothen = @(im) imfilter(im, [1 2 1]'/4, 'symmetric');
            taper = 20;  % the larger the number the thinner the taper
            mask = atan(taper*hanning(sz(1)))*atan(taper*hanning(sz(2)))'/atan(taper)^2;
            
            skipFrames = max(0, min(2000, nframes-5000));
            accumFrames = min(5000, nframes-skipFrames);
            for islice = 1:nslices
                key.slice = islice;

                fprintf('Slice %d of %d\n', islice, nslices)
                fprintf 'Creating template... '
                snippet = getFrames(islice, skipFrames+(1:accumFrames));
                rms = @(img) sqrt(sum(sum(img.^2,1),2));
                template = smoothen(mean(snippet, 5)); 
                for i=1:4
                    corr = bsxfun(@rdivide, mean(mean(bsxfun(@times, snippet, template), 1), 2)./rms(snippet),rms(template));
                    select = bsxfun(@gt, corr, quantile(corr, 0.75, 5));
                    template = smoothen(bsxfun(@rdivide, sum(bsxfun(@times, snippet, select),5), sum(select, 5)));
                end
                key.template = single(template);
                
                templateMean = mean(template(:));
                template = mask.*sharpen(mask.*(template - templateMean));
                ftemplate = conj(fft2(double(template)));
                nframes = reader.nframes;
                xy = nan(2,nframes);

                disp 'Aligning...'
                avgFrame = 0;
                for iframe = 1:nframes
                    if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                        fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                    end
                    frame = getFrames(islice, iframe);
                    if mean(frame(:))-anscombe(0) < 0.25*(templateMean-anscombe(0))
                        % do not attempt motion correction for dark frames
                        x = nan;
                        y = nan;
                    else
                        [x, y] = ne7.ip.measureShift(fft2(frame).*ftemplate);
                        avgFrame = avgFrame + ne7.ip.correctMotion(frame, [x;y])/nframes;
                    end
                    xy(:,iframe) = [x;y];
                end
                key.motion_xy = xy;
                
                xy = bsxfun(@times, [dx; dy], xy);  % convert to microns
                xy = bsxfun(@minus, xy, nanmean(xy,2));  % subtract mean
                d = sqrt(sum(xy.^2));   % distance from average position
                key.motion_rms = single(sqrt(nanmean(d.^2)));   % root mean squared distance
                
                self.insert(key)
            end
        end
    end
    
end