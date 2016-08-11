%{
rf.Motion3D (imported) # motion in 3D computed from a ministack
-> rf.MiniStack
-----
anchor_slice : smallint # slice that best matched the average movie frame
motion_x  : longblob   # (um) movement x component
motion_y  : longblob   # (um) movement y component
motion_z  : longblob   # (um) movement z component
%}

classdef Motion3D < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = rf.MiniStack & 'stack is not null';
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            disp 'preparing stack...'
            [stack,zstep] = fetch1(rf.MiniStack & key,'stack','zstep');
            img = fetch1(rf.Align & key, 'green_img');
            stack = conditionStack(stack);
            img = conditionStack(img);
            
            % find image in stack
            for i=1:size(stack,3)
                [xy(i,1),xy(i,2)] = measureShift(fft2(stack(:,:,i)).*conj(fft2(img)));
                mag(i) = sum(sum(applyOffset(stack(:,:,i),xy(i,:)).*img));
            end
            
            % align all frames to the same offset as anchor frame
            [~,anchor] = max(mag);
            for i=1:size(stack,3)
                stack(:,:,i) = applyOffset(stack(:,:,i),xy(anchor,:));
            end
            
            % align stack to itself
            clear xy
            for i=anchor-1:-1:1
                [xy(1),xy(2)] = measureShift(fft2(stack(:,:,i)).*conj(fft2(stack(:,:,i+1))));
                stack(:,:,i) = applyOffset(stack(:,:,i),xy);
            end
            for i=anchor+1:size(stack,3)
                [xy(1),xy(2)] = measureShift(fft2(stack(:,:,i)).*conj(fft2(stack(:,:,i-1))));
                stack(:,:,i) = applyOffset(stack(:,:,i),xy);
            end
            
            [pxWidth,umWidth,pxHeight,umHeight,nFrames] = ...
                fetch1(rf.ScanInfo*rf.Align & key, ...
                'px_width', 'um_width', 'px_height', 'um_height', 'nframes');
            pixelPitch = (umWidth+umHeight)/(pxWidth + pxHeight);
            reader = rf.getReader(key);
            [fillFraction, rasterPhase] = fetch1(rf.Align & key, 'fill_fraction', 'raster_phase');
            blockSize = 500;
            assert(reader.nSlices==1)
            iSlice = 1;
            reader.reset
            xymotion = fetch1(rf.Align & key, 'motion_xy');
            key.motion_x  = single(squeeze(xymotion(1,1,:))*pixelPitch);
            key.motion_y  = single(squeeze(xymotion(2,1,:))*pixelPitch);
            
            xymotion(:,:,end+1) = xymotion(:,:,end);
            
            z = nan(nFrames,1,'single');  % z-movement
            lastPos = 0;
            while ~reader.done
                fprintf('Reading frames %4d-%4d  ',lastPos+1,lastPos+blockSize)
                block = getfield(reader.read(1, iSlice, blockSize),'channel1'); %#ok<GFLD>
                fprintf condition-
                sz = size(block);
                block = rf.Align.correctRaster(block,rasterPhase,fillFraction);
                block = rf.Align.correctMotion(block, xymotion(:,:,lastPos+(1:sz(4))));
                block = reshape(block,pxHeight,pxWidth,[]);
                block = conditionStack(block);
                fprintf -motion-
                z(lastPos+(1:sz(4))) = arrayfun(@(i) maxind(arrayfun(@(j) sum(sum(block(:,:,i).*stack(:,:,j))), 1:size(stack,3))), 1:size(block,3));
                lastPos = lastPos + sz(4);
                disp done
            end
            
            key.anchor_slice = anchor;
            key.motion_z = single(z*zstep);
            self.insert(key)
        end
    end
    
    methods
        function plot(self)
            for key = self.fetch'
                clf
                [times,x,y,z] = fetch1(rf.Sync*rf.Motion3D & key, ...
                    'frame_times', 'motion_x', 'motion_y', 'motion_z');
                plot(times-times(1),bsxfun(@minus,[x y z],mean([x y z])))
                xlabel 'time (s)'
                ylabel '\Deltax, \Deltay, \Deltaz (\mum)'
                set(gcf,'PaperSize', [15 3], 'PaperPosition', [0 0 15 3])
                axis tight
                pos = get(gca,'Position');
                pos(1) = 0.05;  pos(3) = 0.93;
                set(gca,'Position',pos)
                print('-dpdf', sprintf('~/dev/motion-%04u-%02u', key.animal_id, key.scan_idx))
            end
        end
    end
end


function ix = maxind(a)
[~,ix] = max(a);
end


function stack = conditionStack(stack)
% condition images in stack for computing image cross-correlation

% low-pass filter
k = hamming(5);
k = k/sum(k);
stack = imfilter(imfilter(stack,k,'symmetric'),k','symmetric');

% unsharp masking
sigma = 41;  % somewhat arbitrary
k = gausswin(sigma);
k = k/sum(k);
stack = stack - imfilter(imfilter(stack, k, 'symmetric'), k', 'symmetric');

% taper image boundaries
sz = size(stack);
mask = atan(10*hanning(sz(1)))*atan(10*hanning(sz(2)))' /atan(10)^2;
stack = bsxfun(@times, stack, mask);

% normalize
stack = bsxfun(@rdivide, stack, sqrt(sum(sum(stack.^2))));
end


function [x,y,mag] = measureShift(ixcorr)
% measure the shift of img relative to refImg from xcorr = fft2(img).*conj(fft2(refImg))
sz = size(ixcorr);
assert(length(sz)==2 && all(sz(1:2)>=128 & mod(sz(1:2),2)==0), ...
    'image must have even height and width, at least 128 in size')

phase = fftshift(angle(ixcorr));
mag   = fftshift(abs(ixcorr));
center = sz/2+1;
phaseSlope = [0 0];
for r=[10 15 20]
    [x,y] = meshgrid(-r:r,-r:r);
    plane  = phaseSlope(1)*x + phaseSlope(2)*y;
    phase_ = mod(pi + phase(center(1)+(-r:r),center(2)+(-r:r)) - plane, 2*pi) - pi + plane;
    mag_ = mag(center(1)+(-r:r),center(2)+(-r:r));
    mdl = LinearModel.fit([x(:) y(:)], phase_(:), 'Weights', mag_(:));
    phaseSlope = mdl.Coefficients.Estimate(2:3)';
end
x = -phaseSlope(1)*sz(2)/(2*pi);
y = -phaseSlope(2)*sz(1)/(2*pi);
end


function img = applyOffset(img,xy)
assert(ismatrix(img))
sz = size(img);
g = griddedInterpolant(img,'linear','nearest');
img(:,:) = g({(1:sz(1))+xy(2), (1:sz(2))+xy(1)});
end

