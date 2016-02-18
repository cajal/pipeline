%{
pre.AverageFrame (imported) # my newest table
-> pre.AlignMotion
-> pre.Channel
-----
frame  :  longblob    #  motion aligned and computed as q-norm to approximate maximum projection
%}

classdef AverageFrame < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = pre.AlignRaster & pre.AlignMotion
	end

	methods(Access=protected)

		function makeTuples(self, key)
            q = 10;
            tic
            fixRaster = get_fix_raster_fun(pre.AlignRaster & key);
            reader = pre.getReader(key);
            nframes = reader.nframes;
            for islice = 1:reader.nslices
                key.slice = islice;
                fixMotion = get_fix_motion_fun(pre.AlignMotion & key);
                tuple = key;
                for ichannel = 1:reader.nchannels
                    tuple.channel = ichannel;
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