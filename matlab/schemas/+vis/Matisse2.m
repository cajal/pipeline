%{
vis.Matisse2 (manual) # conditions for the moving matisse stimulus
-> vis.Condition
---
-> vis.Matisse2Cache
pre_blank_period            : decimal(5,3)                  # (seconds)
duration                    : decimal(5,3)                  # (seconds)
pattern_width               : smallint                      # pixel size of the resulting pattern
pattern_aspect              : float                         # the aspect ratio of the pattern
pattern_upscale             : tinyint                       # integer upscale factor of the pattern
ori                         : decimal(4,1)                  # degrees. 0=horizontal, then clockwise
outer_ori_delta             : decimal(4,1)                  # degrees. Differerence of outer ori from inner.
coherence                   : decimal(4,1)                  # 1=unoriented noise. pi/ori_coherence = bandwidth of orientations.
aperture_x                  : decimal(4,3)                  # x position of the aperture in units of pattern widths: 0=center, 0.5=right edge
aperture_y                  : decimal(4,3)                  # y position of the aperture in units of pattern widths: 0=center, 0.5/pattern_aspect = bottom edge
aperture_r                  : decimal(4,3)                  # aperture radius expressed in units pattern widths
aperture_transition         : decimal(3,3)                  # aperture transition width
annulus_alpha               : decimal(3,2)                  # aperture annulus alpha
inner_contrast              : decimal(4,3)                  # pattern contrast in inner region
outer_contrast              : decimal(4,3)                  # pattern contrast in outer region
inner_speed                 : float                         # (units/s)  where unit is display width
outer_speed                 : float                         # (units/s)  where unit is display width
second_photodiode=0         : tinyint                       # 1/-1=paint a photodiode white/black patch in the upper right corner
second_photodiode_time=0.0  : decimal(4,1)                  # time delay of the second photodiode relative to the stimulus onset
%}

classdef Matisse2 < dj.Relvar
    
    methods(Static)
        
        function test()
            fps = 60;
            cond.pre_blank_period = 1.0;
            cond.noise_seed = 100;
            cond.pattern_width = 64;
            cond.pattern_upscale = 3;
            cond.duration = 10;
            cond.pattern_aspect = 1.7;
            cond.ori = 30;
            cond.outer_ori_delta = 90;
            cond.coherence = 2.5;
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
            img = vis.Matisse2.make(cond, fps);
            toc
            
            v = VideoWriter('Matisse2', 'MPEG-4');
            v.FrameRate = fps;
            v.Quality = 100;
            open(v)
            writeVideo(v, permute(img, [1 2 4 3]));
            close(v)

        end
        

        function [img, hash] = make(cond, fps)            
            nframes = round(cond.duration*fps);
            hash = dj.DataHash(setfield(...
                rmfield(cond, {'pre_blank_period', 'duration'}), ...
                'add', [nframes, fps]), struct('Format', 'base64'));
            hash = hash(1:20);
            k.cond_hash = hash;
            fprintf .
            m = fetch(vis.Matisse2Cache & k, '*');
            if ~isempty(m)
                assert(isscalar(m))
                img = m.movie;
            else
                img = randn(round(cond.pattern_width/cond.pattern_aspect), cond.pattern_width);
                outer = upscale(img, cond.pattern_upscale, cond.ori + cond.outer_ori_delta, ...
                    cond.coherence, nframes, cond.outer_speed*cond.pattern_upscale*cond.pattern_width/fps);
                inner = upscale(img, cond.pattern_upscale, cond.ori, ...
                    cond.coherence, nframes, cond.inner_speed*cond.pattern_upscale*cond.pattern_width/fps);
                img = aperture(inner*cond.inner_contrast, outer*cond.outer_contrast, ...
                    cond.aperture_x, cond.aperture_y, cond.aperture_r, cond.aperture_transition, cond.annulus_alpha);
                img = uint8(img*256+127.5);
                k.movie = img;
                insert(vis.Matisse2Cache, k);            
            end
        end
    end
end


function img = upscale(img, factor, ori, coherence, nframes, speed)
% Performs fast resizing of the image by the given integer factor with
% gaussian interpolation.
% speed is expressed in pixels per frame

ori_mix = coherence > 1;  % how much of orientation to mix in

% upscale without interpolation
kernel_sigma = factor;
img = upsample(img', factor, round(factor/2))*factor;
img = upsample(img', factor, round(factor/2))*factor;

% interpolate using gaussian kernel with DC gain = 1
sz = size(img);
[fy,fx] = ndgrid(...
    (-floor(sz(1)/2):floor(sz(1)/2-0.5))*2*pi/sz(1), ...
    (-floor(sz(2)/2):floor(sz(2)/2-0.5))*2*pi/sz(2));

fmask = exp(-(fy.^2 + fx.^2)*kernel_sigma.^2/2);

% apply orientation selectivity and orthogonal motion
ori = ori*pi/180-pi/2;
theta = mod(atan2(fx,fy) + ori, 2*pi) - pi/2;
motion = exp(bsxfun(@times, ifftshift(-1j*speed*(cos(ori).*fx + sin(ori).*fy)), reshape(0:nframes-1, 1, 1, nframes)));
fmask = ifftshift(fmask.*(1-ori_mix + ori_mix*hann(theta*coherence)));
img = real(ifft2(bsxfun(@times, motion, fmask.*fft2(img))));

% contrast compensation for the effect of orientation selectivity
img = img*(1 + ori_mix*(sqrt(coherence)-1));
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
img = bsxfun(@times, inner, 1-mask) + bsxfun(@times, outer, mask);
img = bsxfun(@times, img, 1 - annulus_alpha*(abs(r-radius)<transition/2));
end
