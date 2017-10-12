%{
# quantal size in images
-> preprocess.Prepare
-> preprocess.Channel
-> preprocess.Slice
---
min_intensity           : int                           # max value in movie
max_intensity           : int                           # max value in movie

intensities             : longblob                      # intensities for fitting variances
variances               : longblob                      # variances for each bin

zero_level              : int                           # level that corresponds to dark (computed from variance dependence)
quantal_size            : float                # variance slope, corresponds to quantal size
frame                   : longblob             # average frame expressed in quanta/frame/pixel
median_quantum_rate     : float                # (Hz) median value in frame
percentile95_quantum_rate            : float                # 95th percentile in frame
%}


classdef QuantumSize < dj.Imported
    
    properties
        keySource = preprocess.Prepare * preprocess.Channel * ...
            preprocess.Slice & preprocess.PrepareGalvoMotion & (preprocess.PrepareGalvo & 'nframes>8000')
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            disp 'Loading header...'
            reader = preprocess.getGalvoReader(key);
            skipFrames = 2000;
            nframes = 5000;
            movie = double(squeeze(...
                reader(:,:,key.channel,key.slice,skipFrames+(1:min(nframes,reader.nframes)))));
            
            % compute image range
            intensities = round((movie(:,:,1:end-1) + movie(:,:,2:end))/2);
            [counts,bins] = hist(intensities(:), min(intensities(:)):max(intensities(:)));
            ix = find(counts>200);
            key.min_intensity = bins(ix(1));
            key.max_intensity = bins(ix(end));
            clear ix bins
            
            % compute quantal size
            variances = (movie(:,:,1:end-1) - movie(:,:,2:end)).^2/2;
            jx = intensities >= key.min_intensity & intensities <= min(2500, key.max_intensity);
            intensities = intensities(jx);
            variances = variances(jx);
            
            bins = key.min_intensity:max(intensities);
            counts = hist(intensities, bins);
            variances = accumarray(intensities-bins(1)+1, ...
                variances, [length(bins) 1], @mean);
            key.intensities = single(bins(:));
            key.variances = single(variances(:));
                                    
            jx = counts > 250;
            a = robustfit(bins(jx), variances(jx));
            key.quantal_size = a(2);
            key.zero_level = -a(1)/a(2);
            
            % compute average frame
            nframes = 2000;
            frame = mean(double(squeeze(...
                reader(:,:,key.channel,key.slice,skipFrames:max(1,floor((reader.nframes-skipFrames)/nframes)):reader.nframes))),3);
            key.frame = single((frame-key.zero_level)/key.quantal_size);
            key.median_quantum_rate = median(key.frame(:));
            key.percentile95_quantum_rate = quantile(key.frame(:), 0.95);
            
            self.insert(key)
        end
    end
    
end