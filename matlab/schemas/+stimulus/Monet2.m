%{
# Improved Monet stimulus: pink noise with periods of coherent motion
-> stimulus.Condition
-----
fps                         : decimal(6,3)                 # display refresh rate
duration                    : decimal(6,3)                 # (s) trial duration
rng_seed                    : double                       # random number generator seed
pattern_width               : smallint                     # pixel size of the resulting pattern
pattern_aspect              : float                        # the aspect ratio of the pattern
-> stimulus.TempKernel
temp_bandwidth              : decimal(4,2)                 # (Hz) temporal bandwidth of the stimulus
ori_coherence               : decimal(4,2)                 # 1=unoriented noise. pi/ori_coherence = bandwidth of orientations.
ori_fraction                : float                        # fraction of time coherent orientation is on
ori_mix                     : float                        # mixin-coefficient of orientation biased noise
n_dirs                      : smallint                     # number of directions
speed                       : float                        # (units/s)  where unit is display width
directions                  : longblob                     # computed directions of motion in degrees
onsets                      : blob                         # (s) computed
movie                       : longblob                     # (computed) uint8 movie
%}

classdef Monet2 < dj.Manual & stimulus.core.Visual
    
    properties(Constant)
        version = '2'
    end
    
    methods(Static)
        function test
            cond.fps = 30;
            cond.duration = 30;
            cond.rng_seed = 1;
            cond.pattern_width = 80;
            cond.pattern_aspect = 1.7;
            cond.ori_coherence = 1.5;
            cond.ori_fraction = 0.4;
            cond.temp_kernel = 'half-hamming';
            cond.temp_bandwidth = 4;
            cond.n_dirs = 16;
            cond.ori_mix = 1;
            cond.speed = 0.25;
            
            tic
            cond = stimulus.Monet2.make(cond);
            toc
            
            file = fullfile(pwd,'Monet2.mp4');
            fprintf('Saving %s\n', file)
            
            v = VideoWriter(file, 'MPEG-4');
            v.FrameRate = cond.fps;
            v.Quality = 100;
            open(v)
            writeVideo(v, permute(cond.movie, [1 2 4 3]));
            close(v)
        end
        
        function cond = make(cond)
            
            function y = hann(q)
                % hanning window
                y = (0.5 + 0.5*cos(q)).*(abs(q)<pi);
            end

            assert(~verLessThan('matlab','9.1'), ...  % required for no bsxfun
                'Please upgrade MATLAB to R2016b or better')  
            
            
            % apply temporal filter in time domain
            switch cond.temp_kernel
                case 'hamming'
                    semi = round(cond.fps/cond.temp_bandwidth);  % twice the normal width
                    kernel = hamming(semi*2+1);
                case 'half-hamming'                    
                    semi = round(cond.fps/cond.temp_bandwidth*2);  % twice the normal width
                    kernel = hamming(semi*2+1);
                    kernel = kernel(semi+1:end);
                otherwise
                    error 'unknown temporal kernel'
            end
            kernel = kernel/sum(kernel)/sqrt(cond.temp_bandwidth/4);
            
            
            pad = 3;  % image padding to avoid edge correlations
            r = RandStream.create('mt19937ar','NormalTransform', ...
                'Ziggurat', 'Seed', cond.rng_seed);
            m = r.randn(round(cond.pattern_width/cond.pattern_aspect)+pad, ...
                cond.pattern_width+pad, ...
                round(cond.duration*cond.fps) + length(kernel)-1);            
            m = convn(m, permute(kernel, [3 2 1]), 'valid');
            
            % upsample in space
            factor = 3;
            m = upsample(permute(m, [2 1 3]), factor, round(factor/2))*factor;
            m = upsample(permute(m, [2 1 3]), factor, round(factor/2))*factor;
            sz = size(m);
            
            % compute directions 
            period = cond.duration/cond.n_dirs;
            cond.directions = (r.randperm(cond.n_dirs)-1)/cond.n_dirs*360;
            t = (0:sz(3)-1)/cond.fps;
            cond.onsets = ((0.5:cond.n_dirs) - cond.ori_fraction/2)*period;
            direction = nan(size(t));
            for i = 1:length(cond.onsets)
                direction(t > cond.onsets(i) & t<=cond.onsets(i) + period*cond.ori_fraction) = cond.directions(i);
            end
            
            % make interpolation kernel
            [fy,fx] = ndgrid(...
                ifftshift((-floor(sz(1)/2):floor(sz(1)/2-0.5))*2*pi/sz(1)), ...
                ifftshift((-floor(sz(2)/2):floor(sz(2)/2-0.5))*2*pi/sz(2)));
            kernel_sigma = factor;
            finterp = exp(-(fy.^2 + fx.^2)*kernel_sigma.^2/2);
            
            % apply coherent orientation selectivity and orthogonal motion
            m = fft2(m);
            motion = 1;
            speed = cond.pattern_width*cond.speed/cond.fps;  % in pattern widths per frame
            for i = 1:sz(3)
                fmask = motion.*finterp;  % apply motion first so technically motion starts in next frame
                if ~isnan(direction(i))
                    ori = direction(i)*pi/180+pi/2;   % following clock directions
                    theta = mod(atan2(fx,fy) + ori, pi) - pi/2;
                    mix = cond.ori_mix * (cond.ori_coherence > 1);
                    fmask = fmask.*(1-mix + mix*sqrt(cond.ori_coherence).*hann(theta*cond.ori_coherence));
                    motion = motion .* exp(1j*speed*(cos(ori).*fx + sin(ori).*fy));   % negligible error accumulates
                end
                m(:,:,i) = fmask.*m(:,:,i);
            end
            m = real(ifft2(m))*2.5;
            cond.movie = uint8((m(1:floor((end-pad*factor)/2)*2,1:floor((end-pad*factor)/2)*2,:)+0.5)*255);
        end
    end
    
    
    methods
        function showTrial(self, cond)
            % verify that pattern parameters match display settings
            assert(~isempty(self.fps), 'Cannot obtain the refresh rate')
            assert(abs(self.fps - cond.fps)/cond.fps < 0.05, 'incorrect monitor frame rate')
            assert((self.rect(3)/self.rect(4) - cond.pattern_aspect)/cond.pattern_aspect < 0.05, 'incorrect pattern aspect')
                        
            % play movie
            opts.logFlips = true;
            for i=1:size(cond.movie,3)
                tex = Screen('MakeTexture', self.win, cond.movie(:,:,i));
                Screen('DrawTexture',self.win, tex, [], self.rect)
                self.flip(struct('checkDroppedFrames', i>1))
                Screen('close',tex)
            end
        end
    end
end



