%{
pre.AlignMotion (imported) # motion correction
-> pre.AlignRaster
---
motion_xy                   : longblob                      # (pixels) y,x motion correction offsets
motion_rms                  : float                         # (um) stdev of motion
align_times=CURRENT_TIMESTAMP: timestamp                    # automatic
avg_frame=null              : longblob                      # averaged aligned frame
%}

classdef AlignMotion < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.AlignRaster
    end
    
        
    methods(Access=protected)
        
        function makeTuples(self, key)
            tic
            
            [zero_var_intercept, quantal_size] = fetch1(pre.ScanCheck & key, ...
                'min_var_intensity', 'quantal_size');
            
            anscombe = @(img) 2*sqrt(max(0, img-zero_var_intercept)/quantal_size+3/8);   % Anscombe transform
            
            reader = pre.getReader(key, '~/cache');
            fixRaster = get_fix_raster_fun(pre.AlignRaster & key);
            getFrame = @(iframe) fixRaster(anscombe(double(reader(:,:,:,:,iframe))));
            sz = size(reader);
            
            k = gausswin(41); k=k/sum(k);
            sharpen = @(im) im-imfilter(imfilter(im,k,'symmetric'),k','symmetric');
            taper = 40;  % the larger the number the thinner the taper
            mask = atan(taper*hanning(sz(1)))*atan(taper*hanning(sz(2)))'/atan(taper)^2;
            template = fixRaster(self.getTemplate(key));
            template = sharpen(mask.*(template - mean(template(:))));
       
            ftemplate = conj(fft2(double(template)));
            nframes = reader.nframes;
            xy = nan(2,nframes);
            avgFrame = 0;
            meanLevel = mean(template(:));
            frame = getFrame(1);
            nextFrame = frame;
             
            for iframe = 1:nframes
                if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                    fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                end
                if meanLevel>3
                    frame = fixRaster(anscombe(double(reader(:,:,:,:,iframe))));
                else
                    % apply a bit temporal averaging
                    prevFrame = frame;
                    frame = nextFrame;
                    nextFrame = getFrame(min(reader.nframes, iframe+1));
                    frame = prevFrame/4 + frame/2 + nextFrame/4;
                end
                [x, y] = ne7.ip.measureShift(fft2(frame).*ftemplate); 
                xy(:,iframe) = [x;y];
                avgFrame = avgFrame + ne7.ip.correctMotion(frame, [x;y])/nframes;
            end
            % edge-preserving smoothening
            for iter=1:3
                m = medfilt1(xy,5,[],2);  xy=sign(xy-m).*max(0,abs(xy-m)-0.15)+m;
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
    
    
    methods(Static)
        function img = getTemplate(key)
            img = fetch1(pre.ScanCheck & key, 'template');
        end
    end
    
end
