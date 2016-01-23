%{
rf.Locate (imported) # my newest table
-> rf.Sync
-----
response_map :longblob  # map of response amplitudes on the screen
x_map :longblob  # map of screen locations in brain pixels
y_map :longblob  # map of screen locations in brain pixels
%}

classdef Locate < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = rf.Sync & psy.MovingBar
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            disp 'reading bar positions...'
            trialRel = rf.Sync*psy.Trial*psy.MovingBar & key & ...
                'trial_idx between first_trial and last_trial';
            frameTimes = fetch1(rf.Sync & key, 'frame_times');
            assert(all(trialRel.fetchn('start_pos')==-1) && all(trialRel.fetchn('start_pos')==-1))
            assert(all(ismember(trialRel.fetchn('direction'),0:90:359)))
            tau = 1.7; % (s) calcium tau
            xpos = linspace(-1,1,12);
            ypos = linspace(-1,1,12);
            x = zeros(length(frameTimes),length(xpos));
            y = zeros(length(frameTimes),length(ypos));
            for s = trialRel.fetch('flip_times', 'direction')'
                sgn = 1;
                pos = xpos;
                if ismember(s.direction, [0 270])
                    sgn = -1;
                end
                if ismember(s.direction, [0 180])
                    pos = ypos;
                end
                
                % bins corresponding to the position of the bar at flip_times
                for iBin = 1:length(pos) 
                    passTime = interp1(sgn*linspace(-1,1,length(s.flip_times)), s.flip_times, pos(iBin));
                    t = frameTimes - passTime;
                    ix = t>=0 & t<3*tau;
                    t = t(ix)';
                    switch s.direction
                        case {0 180}
                            y(ix,iBin) = y(ix,iBin) + exp(-t/tau);
                        case {90 270}
                            x(ix,iBin) = x(ix,iBin) + exp(-t/tau);
                    end
                end
            end
            
            disp 'reading movie (1st slice only)...'
            reader = rf.getReader(key);
            [fillFraction, rasterPhase] = fetch1(rf.Align & key, 'fill_fraction', 'raster_phase');
            iSlice = 1;  % read only the first slice
            reader.reset
            blockSize = 1000;
            xymotion = fetch1(rf.Align & key, 'motion_xy');
            [width,height,nFrames] = fetch1(rf.Align*rf.ScanInfo & key, ...
                'px_width','px_height','nframes');
            X = nan(nFrames,width*height,'single');
            lastPos = 0;
            while ~reader.done
                block = getfield(reader.read(1, iSlice, blockSize),'channel1'); %#ok<GFLD>
                sz = size(block);
                xy = xymotion(:,:,1:sz(4));
                xymotion(:,:,1:size(block,4)) = [];
                % block = ne7.ip.correctRaster(block,rasterPhase,fillFraction);
                % block = ne7.ip.correctMotion(block, xy);
                X(lastPos+(1:sz(4)),:) = reshape(block,[],sz(4))';
                lastPos = lastPos + sz(4);
                fprintf('frame %4d\n',lastPos);
            end
            clear block
            
            x = bsxfun(@minus, x, mean(x));
            y = bsxfun(@minus, y, mean(y));
            X = bsxfun(@minus, X, mean(X));
            
            xmap = pinv(x)*X;
            ymap = pinv(y)*X;
            
            figure
            subplot 221
            imagesc(reshape(angle(exp(1i*pi*xpos/2)*xmap)/pi, height, width),[-1 1])
            axis image

            subplot 222
            imagesc(reshape(angle(exp(1i*pi*pos/2)*ymap)/pi, height, width),[-1 1])
            axis image
            
            subplot(2,2,3:4)
            m = mean(ymap,2)*mean(xmap,2)';
            imagesc(m,max(abs(m(:)))*[-1 1]);
            axis image
            
            suptitle(sprintf('Site %d', fetch1(rf.Scan & key, 'site')))
            
            drawnow
        end
    end
end