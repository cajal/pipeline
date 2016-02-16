%{
pre.ScanCheck (imported) # average frame before image corrections
-> pre.ScanInfo
-> pre.Channel
---
avg_frame                   : longblob                      # raw average frame
min_intensity               : int                           # min value in movie
max_intensity               : int                           # max value in movie
min_var_intensity           : int                           # signal level with minimal noise variance, presumed zero
zero_var_intercept          : int                           # the level at which
quantal_size                : float                         # variance slope, corresponds to quantal size
pixel_intensities           : longblob                      # bins used for fitting
half_mean_diff_squared      : longblob                      # measured frame-to-frame variance for each intensity bin
template                    : longblob                      # alignment template after anscombe transform
%}

classdef ScanCheck < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = pre.ScanInfo
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            disp 'Loading header...'
            reader = pre.getReader(key);
            sz = size(reader);
            skipFrames = 500;
            nframes = min(reader.nframes-skipFrames, round(4000/reader.nslices));
            for channel = 1:reader.nchannels
                key.channel = reader.channels(channel);
                fprintf('Channel %d\n', key.channel)
                tic, step = 0;
                movie = single(reshape(reader(:,:,channel,:,skipFrames + (1:nframes)), sz(1), sz(2), sz(4), nframes));
                step = step + 1; fprintf('Step %d Elapsed time %g s\n', step, toc)
                key.avg_frame = single(mean(movie,4));
                
                % compute image range
                step = step + 1; fprintf('Step %d Elapsed time %g s\n', step, toc)
                intensity = round((movie(:,:,:,1:end-1) + movie(:,:,:,2:end))/2);
                [counts,bins] = hist(intensity(:), min(intensity(:)):max(intensity(:)));
                ix = find(counts>100);
                key.min_intensity = bins(ix(1));
                key.max_intensity = bins(ix(end));
                clear ix bins
                
                % compute quantal size
                step = step + 1; fprintf('Step %d Elapsed time %g s\n', step, toc)
                half_diff_squared = (movie(:,:,:,1:end-1) - movie(:,:,:,2:end)).^2/2;
                jx = intensity >= key.min_intensity & intensity <= key.max_intensity;
                intensity = intensity(jx);
                half_diff_squared = half_diff_squared(jx);
                
                step = step + 1; fprintf('Step %d Elapsed time %g s\n', step, toc)
                bins = key.min_intensity:key.max_intensity;
                counts = hist(intensity, bins);
                half_diff_squared = accumarray(intensity-bins(1)+1, ...
                    half_diff_squared, [length(bins) 1], @mean);
                
                step = step + 1; fprintf('Step %d Elapsed time %g s\n', step, toc)
                kx = counts>100 & bins < 4000;
                clear counts
                bins = bins(kx);
                half_diff_squared = half_diff_squared(kx);
                
                [~, j] = min(half_diff_squared);
                key.min_var_intensity = bins(j);
                key.pixel_intensities = single(bins(:));
                key.half_mean_diff_squared = single(half_diff_squared(:));
                
                step = step + 1; fprintf('Step %d Elapsed time %g s\n', step, toc)
                range = bins > key.min_var_intensity;
                a = robustfit(bins(range), half_diff_squared(range));
                key.quantal_size = a(2);
                key.zero_var_intercept = -a(1)/a(2);
               
                % compute template
                step = step + 1; fprintf('Step %d Elapsed time %g s\n', step, toc)             
                anscombe = @(img) 2*sqrt(max(0, img-key.zero_var_intercept)/key.quantal_size+3/8);   % Anscombe transform
                rms = @(img) sqrt(sum(sum(img.^2,1),2));
                movie = anscombe(max(key.zero_var_intercept, movie));
                
                template = mean(movie,4);
                for i=1:4
                    corr = bsxfun(@rdivide, mean(mean(bsxfun(@times, movie, template), 1), 2)./rms(movie),rms(template));
                    select = bsxfun(@gt, corr, quantile(corr, 0.75, 4));
                    template = bsxfun(@rdivide, sum(bsxfun(@times, movie, select),4), sum(select, 4));
                end
                key.template = template;
                
                step = step + 1; fprintf('Step %d Elapsed time %g s\n', step, toc)
                self.insert(key)
                disp done
            end
        end
    end
    
end
