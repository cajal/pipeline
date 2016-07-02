%{
preprocess.PrepareGalvoAverageFrame (imported) # average frame for each slice and channel after corrections
-> preprocess.PrepareGalvoMotion
-> preprocess.Channel
---
frame                       : longblob                      # average frame ater Anscombe, max-weighting,
%}


classdef PrepareGalvoAverageFrame < dj.Relvar
	methods
        
        function save(self)
            for key = fetch(preprocess.PrepareGalvoMotion & self)'
                frames = fetchn(preprocess.PrepareGalvoAverageFrame & key, 'frame', 'ORDER BY frame DESC');
                path = sprintf('~/Desktop/frames/frame%05u-%05u-%u.png', ...
                    key.animal_id, key.scan_idx, key.slice);
                frames = cellfun(@(f) sqrt((f-min(f(:)))/(max(f(:))-min(f(:)))+0.01)-0.1, frames, 'uni', false);
                frames = cat(3,frames{:});
                if size(frames,3) == 2
                    frames(:,:,3)=0;
                end
                imwrite(frames, path)
            end
        end

		function makeTuples(self, key, reader)
            q = 6;
            [nframes, nslices, nchannels] = fetch1(preprocess.PrepareGalvo & key, ...
                'nframes', 'nslices', 'nchannels');
            % average frame
            fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
            for islice = 1:nslices
                key.slice = islice;
                fixMotion = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
                for ichannel = 1:nchannels
                    tic
                    tuple = key;
                    tuple.channel = ichannel;
                    fprintf('Averaging slice %d/%d channel %d\n', islice, nslices, ichannel) 
                    frame = 0;
                    for iframe = 1:nframes
                        frame = frame + max(0,fixMotion(fixRaster(reader(:,:,ichannel, islice, iframe)), iframe)).^q;
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