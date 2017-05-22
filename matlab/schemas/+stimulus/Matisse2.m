%{
# Matisse stimulus:  Moving white noise with orientation coherence with inner and outer regions
-> stimulus.Condition
---
fps                    :decimal(6,3)  #  (Hz) monitor refresh rate
noise_seed             :double        #  RNG seed
pre_blank_period       :decimal(5,3)  #  (seconds)
duration               :decimal(5,3)  #  (seconds)
pattern_width          :smallint      #  pixel size of the resulting pattern
pattern_aspect         :float         #  the aspect ratio of the pattern
ori                    :decimal(4,1)  #  degrees. 0=horizontal, then clockwise
outer_ori_delta        :decimal(4,1)  #  degrees. Differerence of outer ori from inner.
coherence              :decimal(4,1)  #  1=unoriented noise. pi/ori_coherence = bandwidth of orientations.
aperture_x             :decimal(4,3)  #  x position of the aperture in units of pattern widths: 0=center, 0.5=right edge
aperture_y             :decimal(4,3)  #  y position of the aperture in units of pattern widths: 0=center, 0.5/pattern_aspect = bottom edge
aperture_r             :decimal(4,3)  #  aperture radius expressed in units pattern widths
aperture_transition    :decimal(3,3)  #  aperture transition width
annulus_alpha          :decimal(3,2)  #  aperture annulus alpha
inner_contrast         :decimal(4,3)  #  pattern contrast in inner region
outer_contrast         :decimal(4,3)  #  pattern contrast in outer region
inner_speed            :float         # (units/s)  where unit is display width
outer_speed            :float         # (units/s)  where unit is display width
movie                  :longblob      # computed uint8 movie
%}

classdef Matisse2 < dj.Manual & stimulus.core.Visual
    
    properties(Constant)
        version = '1'
    end
    
    methods(Static)
        
        function test
            cond.fps = 60;
            cond.pre_blank_period = 1.0;
            cond.noise_seed = 100;
            cond.pattern_width = 64;
            cond.duration = 1;
            cond.pattern_aspect = 1.7;
            cond.ori = 270;
            cond.outer_ori_delta = 135;
            cond.coherence = 1.5;
            cond.aperture_x = 0.2;
            cond.aperture_y = 0.1;
            cond.aperture_r = 0.2;
            cond.aperture_transition = 0.1;
            cond.annulus_alpha = 0.0;
            cond.outer_contrast = 1;
            cond.inner_contrast = 1;
            cond.outer_speed = 0.2;
            cond.inner_speed = 0.2;
            tic
            cond = stimulus.Matisse2.make(cond);
            toc
            file = fullfile(pwd, 'Matisse2');
            fprintf('saving %s\n', file)
            v = VideoWriter(file, 'MPEG-4');
            v.FrameRate = cond.fps;
            v.Quality = 100;
            open(v)
            writeVideo(v, permute(cond.movie, [1 2 4 3]));
            close(v)
        end
        
        
        function cond = make(cond)
            assert(~verLessThan('matlab','9.1'), 'Please upgrade MATLAB to R2016b or better')  % required for no bsxfun
            nframes = round(cond.duration*cond.fps);
            r = RandStream.create('mt19937ar','NormalTransform', ...
                'Ziggurat', 'Seed', cond.noise_seed); 
            img = r.randn(round(cond.pattern_width/cond.pattern_aspect), cond.pattern_width);
            outer = upscale(img, cond.ori + cond.outer_ori_delta, ...
                cond.coherence, nframes, cond.outer_speed/cond.fps);
            inner = upscale(img, cond.ori, ...
                cond.coherence, nframes, cond.inner_speed/cond.fps);
            img = aperture(inner*cond.inner_contrast, outer*cond.outer_contrast, ...
                cond.aperture_x, cond.aperture_y, cond.aperture_r, cond.aperture_transition, cond.annulus_alpha);
            cond.movie = uint8((img*0.7+0.5)*255);
        end
    end
    
    methods
        function showTrial(self, cond)
            % verify that pattern parameters match display settings
            assert(~isempty(self.fps), 'Cannot obtain the refresh rate')
            assert(abs(self.fps - cond.fps)/cond.fps < 0.05, 'incorrect monitor frame rate')
            assert((self.rect(3)/self.rect(4) - cond.pattern_aspect)/cond.pattern_aspect < 0.05, 'incorrect pattern aspect')
            
            % blank the screen if there is a blanking period
            if cond.pre_blank_period>0
                opts.logFlips = false;
                self.flip(struct('checkDroppedFrames', false))
                WaitSecs(cond.pre_blank_period);
            end
            
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


function img = upscale(img, ori, coherence, nframes, speed)
% Performs fast resizing of the image by the given integer factor with
% gaussian interpolation.
% speed is patten widths per frame

ori_mix = coherence > 1;  % how much of orientation to mix in
factor = 3;

% upscale without interpolation
kernel_sigma = factor;
img = upsample(img', factor, round(factor/2))*factor;
img = upsample(img', factor, round(factor/2))*factor;

% interpolate using gaussian kernel with DC gain = 1
sz = size(img);
[fy,fx] = ndgrid(...
    ifftshift((-floor(sz(1)/2):floor(sz(1)/2-0.5))*2*pi/sz(1)), ...
    ifftshift((-floor(sz(2)/2):floor(sz(2)/2-0.5))*2*pi/sz(2)));
speed = speed*sz(2);  % convert to pixels per frame

fmask = exp(-(fy.^2 + fx.^2)*kernel_sigma.^2/2);

% apply orientation selectivity and orthogonal motion
ori = ori*pi/180+pi/2;   % following clock directions
theta = mod(atan2(fx,fy) + ori, pi) - pi/2;
motion = exp(1j*speed*(cos(ori).*fx + sin(ori).*fy) .* reshape(0:nframes-1, 1, 1, nframes));
fmask = fmask.*(1-ori_mix + sqrt(coherence)*ori_mix*hann(theta*coherence));
img = real(ifft2(motion .* fmask .* fft2(img)));

end


function y = hann(q)
% circuar hanning mask with symmetric opposite lobes
y = (0.5 + 0.5*cos(q)).*(abs(q)<pi);
end


function img = aperture(inner, outer, x, y, radius, transition, annulus_alpha)
% add aperture and annulus
sz = size(inner);
aspect = sz(1)/sz(2);
[y, x] = ndgrid(linspace(-aspect/2,aspect/2,sz(1))-y, linspace(-.5, .5, sz(2))-x);
r = sqrt(y.*y + x.*x);
mask = 1./(1 + exp(-(r-radius)/(transition/4)));
img = inner.*(1-mask) + outer.*mask;
img = img .* (1 - annulus_alpha*(abs(r-radius)<transition/2));
end