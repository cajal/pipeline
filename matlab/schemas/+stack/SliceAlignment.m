%{
stack.SliceAlignment (imported) # my newest table
-> preprocess.PrepareGalvoAverageFrame
-> experiment.Stack
-----
score:   float      # cross correlation score
depth:   float      # depth in the stack in microns
x_shift: float      # shift in x direction in microns
y_shift: float      # shift in y direction in microns
%}

classdef SliceAlignment < dj.Relvar & dj.AutoPopulate

	properties
		popRel = pro(experiment.Scan) * pro(experiment.Stack & stack.StackMotion) * preprocess.Channel & preprocess.PrepareGalvoAverageFrame;
    end
    
    methods
        function plotSlices(self, channel)
            if nargin == 2
                cond = sprintf('channel = %d', channel);
            else
                cond = [];
            end

            session = fetch(experiment.Session & self);
            assert(length(session) == 1, 'This can only plot a single session at a time');
            scans = fetch(experiment.Scan & session);
            cvals = lines(length(scans));
            
            figure;
            hold on;
            for idxScan = 1:length(scans)
                cbase = cvals(idxScan, :);
                slices = fetch(stack.SliceAlignment & scans(idxScan) & cond, '*');
                for idxSlice = 1:length(slices)
                    slice = slices(idxSlice);

                    c = max(cbase - (idxSlice - 1) * 0.05 * [1,1,1],0);
                    
                    plot([0, 1], [1, 1] * slice.depth, 'color', c);
                    text(0.1, slice.depth + 1, sprintf('Depth: %dum (animal %d session %d scan %d slice %d)', ...
                        slice.depth, slice.animal_id, slice.session, slice.scan_idx, slice.slice));
                end
            end
        end 
    end

	methods(Access=protected)
		function makeTuples(self, key)
            startTime = tic;
            
            scanInfo = fetch(preprocess.PrepareGalvo & key, '*');
            
            fprintf('Loading reader...'); tic;
            reader = stack.getStackReader(key);
            fprintf('(completed in %.2f seconds)\n',toc);
            
            % will work only if the zoom level is the same for the scans
            % and the stack
            assert(abs(reader.zoom - scanInfo.zoom) < eps(scanInfo.zoom));
            scanSize = [scanInfo.px_height scanInfo.px_width];
            stackSize = size(reader);
            
            targetSize = max([scanSize; stackSize(1:2)]);
            pxPitch = [scanInfo.um_height, scanInfo.um_height] ./ targetSize;
            
            fprintf('Loading structural stack...'); tic;
            stk = stack.loadCorrectedStack(key);
            avgStack = squeeze(mean(stk(:, :, key.channel, :, :), 5));
            clear stk
            fprintf('(completed in %.2f seconds)\n',toc);
            
            fprintf('Normalizing structural stack...'); tic;
            normStack = normalizeImages(avgStack, targetSize);
            fprintf('(completed in %.2f seconds)\n',toc);
            
            
            % now align each frame, one at a time
            avgFrameKeys = fetch(preprocess.PrepareGalvoAverageFrame & key);
            for idxFrame = 1:length(avgFrameKeys)
                fprintf('Aligning slice %d...', idxFrame); tic;
                tuple = avgFrameKeys(idxFrame);
                
                % normalize the frame
                avgFrame = fetch1(preprocess.PrepareGalvoAverageFrame & tuple, 'frame');
                normFrame = normalizeImages(avgFrame, targetSize);
                
                % cross correlation to find best maching indicies
                sliceLoc = findImageInStack(normFrame, normStack);
                
                % convert pixcels and slice number into microns
                tuple.stack_idx = key.stack_idx;
                tuple.score = sliceLoc.score;
                tuple.depth = (sliceLoc.depth - 1) * reader.slice_pitch;
                tuple.y_shift = sliceLoc.shift(1) * pxPitch(1);
                tuple.x_shift = sliceLoc.shift(2) * pxPitch(2);
                
                self.insert(tuple);
                fprintf('(completed in %.2f seconds)\n',toc);
            end
            
            fprintf('Overall completed in %.2f seconds\n',toc(startTime));
        end
    end
end

function imgLoc = findImageInStack(img, stack)
% Given a stack of images and an image,
% finds the best matching x,y shift and the depth
% within the stack
    maxScore = 0;
    maxDepth = 1;
    maxPos = 0;
    for idxDepth = 1:size(stack, 3)
        template = stack(:,:,idxDepth);

        % compute cross correlation via Fourier domains
        xcorr = ifft2(fft2(img) .* conj(fft2(template)));
        [score, pos] = max(xcorr(:));
        if score > maxScore
            maxScore = score;
            maxDepth = idxDepth;
            maxPos = pos;
        end
    end
    sz = size(img);
    [y, x] = ind2sub(sz, maxPos);
    yh = sz(1)/2; y = mod(y+yh, sz(1)) - yh;
    xh = sz(2)/2; x = mod(x+xh, sz(2)) - xh;

    % record the result
    imgLoc.score = maxScore;
    imgLoc.depth = maxDepth;
    imgLoc.shift = [y, x];
end

function normalizedImages = normalizeImages(images, targetSize)
% Normalize each image such that the mean is 0 and L2 norm of the
% image is 1 (same thing as variance of 1). If images is 3D, assumes that
% the 3rd index is the image number and applies normalization on each
% image separately.
    sz = size(images);
    if nargin < 2
        targetSize = sz(1:2);
    end
    resize = any(targetSize ~= sz(1:2));
    
    if ndims(images) == 2
        N = 1;
    else
        N = size(images, 3);
    end
       
    normalizedImages = zeros([targetSize, N]);
    for idx = 1:N
        img = images(:, :, idx);
        if resize
            img = imresize(img, targetSize);
        end
        img = img - mean(img(:));
        normalizedImages(:, :, idx) = img ./ norm(img(:));
    end
    normalizedImages = squeeze(normalizedImages);
end