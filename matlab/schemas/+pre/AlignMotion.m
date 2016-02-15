%{
pre.AlignMotion (imported) # motion correction
-> pre.AlignRaster
-> pre.Slice
---
-> pre.Channel
motion_xy                   : longblob                      # (pixels) y,x motion correction offsets
motion_rms                  : float                         # (um) stdev of motion
align_times=CURRENT_TIMESTAMP: timestamp                    # automatic
avg_frame=null              : longblob                      # averaged aligned frame
INDEX(animal_id,session,scan_idx,channel)
%}

classdef AlignMotion < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.AlignRaster & pre.ScanCheck
    end
    
    
    methods
        function fun = get_fix_motion_fun(self)
            xy = self.fetch1('motion_xy');
            fun = @(frame, i) ne7.ip.correctMotion(frame, xy(:,i));
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
            
            reader = pre.getReader(key, '~/cache');
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
                key.avg_frame=avgFrame;
                
                self.insert(key)
            end
        end
    end
    
end



function j = argmax(r)
[~,j] = max(r);
end
