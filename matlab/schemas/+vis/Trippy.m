%{
vis.Trippy (manual) # randomized curvy dynamic gratings
-> vis.Condition
---
version                     : tinyint                       # trippy version
rng_seed                    : double                        # random number generate seed
packed_phase_movie          : longblob                      # phase movie before spatial and temporal interpolation
luminance                   : float                         # (cd/m^2)
contrast                    : float                         # michelson contrast
tex_ydim                    : smallint                      # (pixels) texture dimension
tex_xdim                    : smallint                      # (pixels) texture dimension
duration                    : float                         # (s) trial duration
frame_downsample            : tinyint                       # 1=60 fps, 2=30 fps, 3=20 fps, 4=15 fps, etc
xnodes                      : tinyint                       # x dimension of low-res phase movie
ynodes                      : tinyint                       # y dimension of low-res phase movie
up_factor                   : tinyint                       # spatial upscale factor
temp_freq                   : float                         # (Hz) temporal frequency if the phase pattern were static
temp_kernel_length          : smallint                      # length of Hanning kernel used for temporal filter. Controls the rate of change of the phase pattern.
spatial_freq                : float                         # (cy/degree) approximate max. The actual frequencies may be higher.
%}


classdef Trippy < dj.Relvar
    
    properties(Constant)
        version = 1
    end
    
    methods(Static)
        
        function phase = make_packed_phase_movie(cond, fps, degxy)
            % Make compressed phase movie.
            r = RandStream.create('mt19937ar','Seed', cond.rng_seed);
            nframes = ceil(cond.duration*fps);
            n = [cond.ynodes cond.xnodes];
            k = cond.temp_kernel_length;
            assert(k>=3 && mod(k,2)==1)
            k2 = ceil(k/4);
            compensator = 8.0;
            scale = compensator*cond.up_factor*cond.spatial_freq*degxy(1)/cond.tex_xdim;
            phase = scale*r.rand(ceil((nframes+k-1)/k2),prod(n));
        end
        
        
        function phase = interp_time(phase, cond, fps)
            nframes = ceil(cond.duration*fps);
            
            % lowpass in time
            k = cond.temp_kernel_length;
            assert(k>=3 && mod(k,2)==1)
            k2 = ceil(k/4);
            phase = upsample(phase, k2);
            tempKernel = hanning(k);
            tempKernel = k2/sum(tempKernel)*tempKernel;
            phase=conv2(phase,tempKernel,'valid');  % lowpass in time
            phase = phase(1:nframes,:);
            
            % add motion
            phase=bsxfun(@plus, phase, (1:nframes)'/fps*cond.temp_freq);  % add motion
        end
        
        
        function movie = interp_space(phase, cond)
            % upscale to full size
            n = [cond.ynodes cond.xnodes];
            f = cond.up_factor;
            movie = zeros(cond.ynodes*f, cond.xnodes*f, size(phase,1));
            for i=1:size(phase,1)
                movie(:,:,i) = vis.Trippy.frozen_upscale(reshape(phase(i,:),n),f);
            end
            % crop to screen size
            movie = movie(1:cond.tex_ydim, 1:cond.tex_xdim, :);
        end
        
        
        function img = frozen_upscale(img, factor)
            % Performs fast resizing of the image by the given integer factor with
            % gaussian interpolation.
            % Never modify this function. Ever. It was used to generate Version 1 trippy movies.
            % Frozen on 2015-12-30. No changes are allowed ever.
            
            for i=1:2
                img = upsample(img', factor, round(factor/2));
                L = size(img,1);
                k = gausswin(L,sqrt(0.5)*L/factor);
                k = ifftshift(factor/sum(k)*k);
                img = real(ifft(bsxfun(@times, fft(img), fft(k))));
            end
        end
        
        
        function test()
            
            cond.rng_seed = 1;
            cond.tex_ydim = 90;  %
            cond.tex_xdim = 160;  %
            cond.duration = 60;   % (s) trial duration
            cond.xnodes = 8;     % x dimension of low-res phase movie
            cond.ynodes = 6;      % y dimension of low-res phase movie
            cond.up_factor = 24;  % upscale factor from low-res to texture dimensions
            cond.temp_freq = 2.5;   % (Hz) temporal frequency if the phase pattern were static
            cond.temp_kernel_length = 61;  % length of Hanning kernel used for temporal filter. Controls the rate of change of the phase pattern.
            cond.spatial_freq = 0.06;  % (cy/degree) approximate max. Actual frequency spectrum ranges propoprtionally.
            
            fps = 60;
            screenSize = [160 90];  % degrees
            phase = vis.Trippy.make_packed_phase_movie(cond, fps, screenSize);
            
            phase = vis.Trippy.interp_time(phase, cond, fps);
            phase = vis.Trippy.interp_space(phase, cond);
            movie = cos(2*pi*phase)/2+1/2;
            
            % plot statistics
            degPerPixel = screenSize(2)/cond.tex_ydim;
            [gx, gy] = gradient(phase, degPerPixel);
            spatial_freq = sqrt(gx.^2 + gy.^2);
            temp_freq = (phase(:,:,[2:end end]) - phase(:,:,[1 1:end-1]))*fps/2;
            subplot 121,  hist(spatial_freq(:),300); box off, xlabel 'log spatial frequencies (cy/degree)'
            xlim([0 0.2]), grid on
            set(gca, 'YTickLabel', [])
            subplot 122,  hist(abs(temp_freq(:)),300);   box off,  xlabel 'temporal frequencies (Hz)'
            xlim([0 15]), grid on
            set(gca, 'YTickLabel', [])
            set(gcf, 'PaperSize', [8 3], 'PaperPosition', [0 0 8 3])
            print -dpng ~/Desktop/trippy-stats
            
            % save movie
            v = VideoWriter('~/Desktop/trippy', 'MPEG-4');
            v.FrameRate = fps;
            v.Quality = 100;
            open(v)
            writeVideo(v, permute(movie, [1 2 4 3]));
            close(v)
        end
    end
end