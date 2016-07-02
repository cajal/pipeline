%{
preprocess.PrepareGalvoAverageFrame (imported) # average frame for each slice and channel after corrections
-> preprocess.PrepareGalvoMotion
-> preprocess.Channel
---
frame                       : longblob                      # average frame ater Anscombe, max-weighting,
%}


classdef PrepareGalvoAverageFrame < dj.Relvar
	methods

		function makeTuples(self, key, reader)
            q = 6;
            [nframes, nslices, nchannels] = fetch1(preprocess.PrepareGalvo, ...
                'nframes', 'nslices', 'nchannels');
            % average frame
            for islice = 1:nslices
                key.slice = islice;
                fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
                tuple = key;
                for ichannel = 1:nchannels
                    tic
                    key.channel = ichannel;
                    print('Averaging slice %d/%d channel %d\n', islice, nslices, ichannel) 
                    tuple.channel = ichannel;
                    fixMotion = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
                    frame = 0;
                    for iframe = 1:nframes
                        frame = frame + (fixMotion(fixRaster(reader(:,:,ichannel, islice, iframe)), iframe)).^q;
                        if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                            fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                        end
                    end
                    tuple.frame = single((frame/nframes).^(1/q));
                    self.insert(tuple)
                end
            end
		end
    end
end