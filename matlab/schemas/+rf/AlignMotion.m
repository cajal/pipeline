%{
rf.AlignMotion (imported) # motion correction based on the first channel
-> rf.AlignRaster
---
motion_xy                   : longblob                      # (pixels) y,x motion correction offsets
motion_rms                  : float                         # (um) stdev of motion
template                    : longblob                      # template image resulting from alignment
timestamp=CURRENT_TIMESTAMP : timestamp                     # automatic
%}

classdef AlignMotion < dj.Relvar & dj.AutoPopulate
    
    properties(Constant)
        popRel = rf.AlignRaster
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            [fillFraction, rasterPhase, nslices] = fetch1(...
                rf.AlignRaster*rf.ScanInfo & key, ...
                'fill_fraction', 'raster_phase', 'nslices');
            [avgFrame, level0, quantalSize] = fetch1(rf.ScanCheck & key, ...
                'avg_frame', 'min_var_intensity', 'quantal_size');
            assert(nslices == 1, 'This module has not been tested for multislice scans')
            
            reader = rf.getReader(key);
            nframes = reader.hdr.acqNumFrames;
            fixRaster = @(x) ne7.ip.correctRaster(x, rasterPhase, fillFraction);
            fixMotion = @(X, x, y) ne7.ip.correctMotion(X, permute([x y], [2 3 1]));
            anscombe = @(img) 2*sqrt(max(0, img-level0)/quantalSize+3/8);   % Anscombe transform
            compress = @(x) x./sqrt(1+x.^2);   % suppresses outliers
            k = gausswin(21); k=k/sum(k);
            sharpen = @(im) im-imfilter(imfilter(im,k,'symmetric'),k','symmetric');
            
            % get initial template
            x = zeros(nframes,1);
            y = zeros(nframes,1);
            sz = size(avgFrame);
            taper = 20;  % the larger the number the thinner the taper
            mask = atan(taper*hanning(sz(1)))*atan(taper*hanning(sz(2)))'/atan(taper)^2;
            newS = mask.*anscombe(fixRaster(avgFrame));
            presmooth = [3 8 10 8 3]/32;
            kix = (1:length(presmooth))-ceil(length(presmooth)/2);
            beta = [0 0.2 0.5];
            for pass = 1:3
                fprintf('pass %d\n',pass)
                reader.reset
                tic
                S = mask.*sharpen(newS);
                fS = fft2(S);
                newS = 0;
                for iframe=1:nframes
                    if ismember(iframe,[10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                        fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                    end
                    frame = anscombe(fixRaster(getfield(reader.read(1,1,1), 'channel1'))); %#ok<GFLD>
                    x_ = presmooth*x(max(1,min(end,kix+iframe)));
                    y_ = presmooth*y(max(1,min(end,kix+iframe)));
                    out = ne7.ip.dftregistration(fft2(sharpen(fixMotion(frame, x_, y_))), fS, 10);
                    x(iframe) = x_ + sign(out(4))*max(0, min(5, abs(out(4))-beta(pass)));
                    y(iframe) = y_ + sign(out(3))*max(0, min(5, abs(out(3))-beta(pass)));
                    newS = newS + compress(fixMotion(frame, x(iframe), y(iframe))-newS)/iframe;
                end
                figure;  imagesc(newS); axis image; colorbar; drawnow
            end
            
            key.motion_xy = permute([x y], [2 3 1]);
            key.motion_rms = sqrt(mean(x.^2 + y.^2));
            key.template = newS;
            self.insert(key)
        end
    end
end
