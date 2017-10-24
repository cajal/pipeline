%{
stack.StackMotion (imported) # motion correction
-> stack.StackInfo
-----
raw_motion_xy               : longblob                      # (pixels) y,x motion correction offsets
motion_xy                   : longblob                      # (pixels) y,x motion correction offsets after detrending
%}

classdef StackMotion < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = stack.StackInfo;
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            
            tic;
            
            %% load and init
            fprintf('Loading stack...');
            reader = stack.getStackReader(key);
            stk = reader(:, :, :, :, :);
            oldStack = stk;
            nslices = reader.nslices;
            nframes = reader.nframes;
            slicePitch = reader.slice_pitch;
            
            if reader.nchannels > 1
                channel = 2; % MICrONS-specific, within-slice alignment works better on channel 2
            else
                channel = 1;
            end
            
            xy = zeros(2,nframes,nslices);
            fprintf('(completed in %.2f seconds)\n',toc);
            
            % set up Anscombe transform
            zero=0;
            quantal_size = 50;  % from prior experience
            anscombe = @(img) 2*sqrt(max(0, img-zero)/quantal_size+3/8);   % Anscombe transform
            
            %% align frames within slices first
            nIter=3;
            for islice = 1:nslices
                for iter = 1:nIter
                    fprintf('Slice %d of %d\n', islice, nslices)
                    fprintf 'Creating template... '
                    snippet = anscombe(double(stk(:,:,channel,islice,:)));
                    [ftemplate,templateMean] = stack.StackMotion.makeTemplate(snippet);
                    for iframe = 1:nframes
                        frame = anscombe(double(stk(:,:,channel,islice,iframe)));
                        if mean(frame(:))-anscombe(0) < 0.25*(templateMean-anscombe(0))
                            % do not attempt motion correction for dark frames
                            x = nan;
                            y = nan;
                        else
                            [x, y] = ne7.ip.measureShift(fft2(frame).*ftemplate);
                            stk(:,:,1,islice,iframe) = int16(ne7.ip.correctMotion(single(stk(:,:,1,islice,iframe)), [x;y]));
                            stk(:,:,2,islice,iframe) = int16(ne7.ip.correctMotion(single(stk(:,:,2,islice,iframe)), [x;y]));
                        end
                        xy(:,iframe,islice) = xy(:,iframe,islice)+[x;y];
                    end
                end
            end
            
            %% then align across slices
            channel=1; % Possibly MICrONS-specific, across-slice alignment works better on channel 1
            
            for islice = 2:nslices
                fprintf('Slice %d of %d\n', islice, nslices)
                fprintf 'Creating template... '
                snippet = anscombe(double(stk(:,:,channel,islice-1,:)));
                [ftemplate,templateMean] = stack.StackMotion.makeTemplate(snippet);
                frame = anscombe(double(mean(stk(:,:,channel,islice,:),5)));
                if mean(frame(:))-anscombe(0) < 0.25*(templateMean-anscombe(0))
                    % do not attempt motion correction for dark frames
                    x = nan;
                    y = nan;
                else
                    [x, y] = ne7.ip.measureShift(fft2(frame).*ftemplate);
                    for iframe=1:nframes
                        stk(:,:,1,islice,iframe) = int16(ne7.ip.correctMotion(single(stk(:,:,1,islice,iframe)), [x;y]));
                        stk(:,:,2,islice,iframe) = int16(ne7.ip.correctMotion(single(stk(:,:,2,islice,iframe)), [x;y]));
                        xy(:,iframe,islice) = xy(:,iframe,islice)+[x;y];
                    end
                end
            end
            
            %% detrend
            key.raw_motion_xy = xy;
            
            k = hamming(2*round(20*nframes/slicePitch)+1);
            k = k/sum(k);
            
            a = xy(1,:,:); b = xy(2,:,:);
            sz = size(a);
            a = a(:) - ne7.dsp.convmirr(a(:),k);
            b = b(:) - ne7.dsp.convmirr(b(:),k);
            xy = cat(1,reshape(a,sz),reshape(b,sz));
            
            key.motion_xy = xy;
            
            %% insert
            self.insert(key);
            fprintf('(completed in %.2f seconds)\n',toc);
        end
    end
    
    methods(Static)
        function [ftemplate,templateMean] = makeTemplate(snippet)
            
            %% Yatsenko magic
            sz = size(snippet);
            k = gausswin(41); k=k/sum(k);
            sharpen = @(im) im-imfilter(imfilter(im,k,'symmetric'),k','symmetric');
            smoothen = @(im) imfilter(im, [1 2 1]'/4, 'symmetric');
            taper = 20;  % the larger the number the thinner the taper
            mask = atan(taper*hanning(sz(1)))*atan(taper*hanning(sz(2)))'/atan(taper)^2;
            rms = @(img) sqrt(sum(sum(img.^2,1),2));
            
            template = smoothen(mean(snippet, 5));
            for i=1:4
                corr = bsxfun(@rdivide, mean(mean(bsxfun(@times, snippet, template), 1), 2)./rms(snippet),rms(template));
                select = bsxfun(@gt, corr, quantile(corr, 0.75, 5));
                template = smoothen(bsxfun(@rdivide, sum(bsxfun(@times, snippet, select),5), sum(select, 5)));
            end
            templateMean = mean(template(:));
            template = mask.*sharpen(mask.*(template - templateMean));
            ftemplate = conj(fft2(double(template)));
            
        end
    end
end

