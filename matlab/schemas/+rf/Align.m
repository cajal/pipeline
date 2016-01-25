%{
rf.Align (imported) # motion correction$
-> rf.ScanInfo
---
nframes                     : int                           # actual number of recorded frames
fill_fraction               : float                         # scan fill fraction (see scanimage)
raster_phase                : float                         # shift of odd vs even raster lines
motion_xy                   : longblob                      # (pixels) y,x motion correction offsets
motion_rms                  : float                         # (um) stdev of motion
xcorr_traces                : longblob                      # peak correlations between frames
green_upper                 : float                         # 99th pecentile of intensity on green channel from the beginning of the movie
raw_green_img=null          : longblob                      # unaligned mean green image
raw_red_img=null            : longblob                      # unaligned mean red image
green_img=null              : longblob                      # aligned mean green image
red_img=null                : longblob                      # aligned mean red image
align_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
%}

classdef Align < dj.Relvar & dj.AutoPopulate
    
    properties(Constant)
        popRel = rf.ScanInfo
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            reader = rf.getReader(key);
            
            info = fetch(rf.ScanInfo & key, '*');
            minFrames = 300;
            assert(info.nframes_requested > minFrames, ...
                'we assume at least %d frames', minFrames)
            %assert(strcmp(reader.hdr.scanMode, 'bidirectional'), ...
            %    'scanning must be bidirectional')
            
            disp 'computing template for motion alignment...'
            reader.read([], [], 60);    % skip some frames (stacks)
            blockSize = 300;   % number of frames used for template
            templateBlock = getfield(reader.read(1,1:info.nslices,blockSize),'channel1'); %#ok<GFLD>
            
            % compute raster correction
            fillFraction = reader.hdr.scanFillFraction;
            if strcmp(reader.hdr.scanMode, 'bidirectional')
                rasterPhase = ne7.ip.computeRasterCorrection(squeeze(mean(templateBlock(:,:,1,:),4)), fillFraction);
            else
                rasterPhase=0;
            end
            
            % compute motion correction template
            if strcmp(reader.hdr.scanMode, 'bidirectional')
                templateBlock = ne7.ip.correctRaster(templateBlock, rasterPhase, fillFraction);
            end
            
            key.green_upper = quantile(templateBlock(:),0.99);
            c = ones(1,blockSize);
            for iter=1:3
                c = reshape(c.^4/sum(c.^4),[1 1 1 blockSize]);
                template = ne7.ip.conditionStack(sum(bsxfun(@times, templateBlock, c),4));
                c = arrayfun(@(i) mean(sum(sum(template.*ne7.ip.conditionStack(templateBlock(:,:,:,i))))), ...
                    1:blockSize);
            end
            
            disp 'aligning motion...'
            reader.reset
            blockSize = 320;
            fTemplate = conj(fft2(template));  % template in frequency domain
            accum = [];
            channels = intersect(1:2, reader.hdr.channelsSave);
            hasRed = length(channels)>=2;
            
            raw = struct('green',0,'red',0);
            img = struct('green',0,'red',0);
            
            while ~reader.done
                % load block
                block = reader.read(channels,1:info.nslices,blockSize);
                sz = size(block.channel1);
                greenBlock = block.channel1;
                
                % compute motion correction
                xymotion = zeros(2,sz(3),sz(4),'single');
                cc = zeros(sz(3),sz(4));
                for iFrame=1:sz(4)
                    if ~mod(iFrame,32), fprintf ., end
                    for iSlice = 1:sz(3)
                        % compute cross-corelation as product in frequency domain
                        if strcmp(reader.hdr.scanMode, 'bidirectional')
                            frame = ne7.ip.correctRaster(greenBlock(:,:,iSlice,iFrame), rasterPhase, fillFraction);
                        else
                            frame = greenBlock(:,:,iSlice,iFrame);
                        end
                        [x,y] = ne7.ip.measureShift(fft2(ne7.ip.conditionStack(frame)).*fTemplate(:,:,iSlice));
                        xymotion(:,iSlice,iFrame) = [x y];
                    end
                end
                
                % accumulate motion correction
                if isempty(accum)
                    accum.xymotion = xymotion;
                    accum.cc = cc;
                else
                    accum.xymotion = cat(3, accum.xymotion,xymotion);
                    accum.cc = cat(2, accum.cc, cc);
                end
                
                if hasRed, redBlock = block.channel2; end
                % display raw averaged block
                doDisplay = false;
                if doDisplay
                    g = reshape(mean(greenBlock,4), size(greenBlock,1),[])/key.green_upper;
                    if hasRed
                        g = cat(3,reshape(mean(redBlock,4), size(redBlock,1),[])/quantile(redBlock(:),0.99),g);
                        g(:,:,3) = 0;
                    end
                    imshow(g)
                    title(sprintf('frames %d-%d', length(accum.cc)-length(cc)+1, length(accum.cc)))
                    drawnow
                end
                % accumuluate raw frames and corrected frame
                raw.green = raw.green + sum(greenBlock,4);
                if strcmp(reader.hdr.scanMode, 'bidirectional')
                    img.green = img.green + sum(ne7.ip.correctMotion(ne7.ip.correctRaster(greenBlock, rasterPhase, fillFraction), xymotion),4);
                else
                    img.green = img.green + sum(ne7.ip.correctMotion(greenBlock, xymotion),4);
                end
                
                if hasRed
                    raw.red = raw.red + sum(redBlock,4);
                    if strcmp(reader.hdr.scanMode, 'bidirectional')
                        img.red = img.red + sum(ne7.ip.correctMotion(ne7.ip.correctRaster(redBlock, rasterPhase, fillFraction), xymotion),4);
                    else
                        img.red = img.red + sum(ne7.ip.correctMotion(redBlock, xymotion),4);
                    end
                end
                
                fprintf(' Aligned %4d frames\n', size(accum.cc,2))
            end
            nFrames = size(accum.cc,2);
            key.fill_fraction = fillFraction;
            key.raster_phase = rasterPhase;
            key.nframes = nFrames;
            key.motion_xy = accum.xymotion;
            key.xcorr_traces = single(accum.cc);
            key.raw_green_img = single(raw.green/nFrames/key.green_upper);
            key.green_img     = single(img.green/nFrames/key.green_upper);
            if hasRed
                key.raw_red_img = single(raw.red/nFrames/key.green_upper);
                key.red_img     = single(img.red/nFrames/key.green_upper);
            end
            pixelPitch = info.um_width / info.px_width;
            key.motion_rms = pixelPitch*sqrt(mean(mean(var(double(accum.xymotion),[],3))));
            self.insert(key)
        end
    end
end
