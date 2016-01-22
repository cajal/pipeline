%{
rf.ScanCheck (imported) # average frame before image corrections
-> rf.ScanInfo
-----
avg_frame :longblob   # raw frame from the first channel
min_var_intensity: int   # signal level with minimal noise variance, presumed zero
zero_var_intercept: int  # the level at which 
quantal_size: float   # variance slope, corresponds to quantal size
pixel_intensities :  longblob   #  bins used for fitting
half_mean_diff_squared: longblob  #  measured frame-to-frame variance for each intensity bin
%}

classdef ScanCheck < dj.Relvar & dj.AutoPopulate

	properties
		popRel = rf.ScanInfo
	end

	methods(Access=protected)

		function makeTuples(self, key)
            reader = rf.getReader(key);
            nframes = reader.hdr.acqNumFrames;
            prev_frame = getfield(reader.read(1,1,1), 'channel1'); %#ok<GFLD>
            avg_frame = prev_frame/nframes;
            offset = 16000;
            binsize = 4;
            pixel_intensities =(1:ceil((offset+32768)/binsize))';
            diff_squared = zeros(size(pixel_intensities));
            counts = zeros(size(pixel_intensities));
            nframes = min(nframes,1e4);
            for i=2:nframes
                assert(~reader.done, 'invalid TIFF file')
                if ismember(i,[10 100 500 1000 5000 nframes]) || mod(i,10000)==0
                    fprintf('Frame %5d/%d\n', i, nframes);
                end
                frame = getfield(reader.read(1,1,1), 'channel1'); %#ok<GFLD>
                avg_frame = avg_frame + frame/nframes;
                ix = round((frame(:)/2 + prev_frame(:)/2 + offset)/binsize);
                assert(all(ix>0))
                counts = counts + hist(ix,pixel_intensities)';
                diff_squared = diff_squared + accumarray(ix,(frame(:)-prev_frame(:)).^2/2, [length(pixel_intensities) 1]);
                prev_frame = frame;
            end
            valid = find(counts>1e3);
            pixel_intensities = pixel_intensities(valid)*binsize-offset;
            counts = counts(valid);
            diff_squared = diff_squared(valid)./counts;
            a = robustfit(pixel_intensities, diff_squared);
            key.quantal_size = a(2);
            key.zero_var_intercept = -a(1)/a(2);
            key.avg_frame = avg_frame;
            key.pixel_intensities = pixel_intensities;
            key.half_mean_diff_squared = diff_squared;
            [~, k] = min(diff_squared);
            key.min_var_intensity = pixel_intensities(k);
            
            self.insert(key)
            disp done
		end
    end

end