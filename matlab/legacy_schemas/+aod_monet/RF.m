%{
aod_monet.RF (computed) # receptive fields from the monet stimulus
-> aodpre.Sync
-> aodpre.Spikes
-----
nbins              : smallint                      # temporal bins
bin_size           : float                         # (ms) temporal bin size
degrees_x          : float                         # degrees along x
degrees_y          : float                         # degrees along y
map             : longblob                      # receptive field map
%}

classdef RF < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = aodpre.ExtractSpikes*(aodpre.Sync & psy.MovingNoise)
    end
    
    
    methods
        function dump(self)
            path = '/Volumes/scratch01/RF-party';
            for key = self.fetch'
                disp(key)
                fullpath = fullfile(path, ...
                    sprintf('%05d-%s', key.mouse_id, ...
                    fetch1(pre.SpikeInference & key, 'short_name')));
                if ~exist(fullpath, 'dir')
                    mkdir(fullpath);
                end
                
                map = fetch1(self & key, 'map');
                mx = max(abs(map(:)));
                map = round(map/mx*31.5 + 32.5);
                cmap = ne7.vis.doppler;
                for i=1:min(size(map,3),6)
                    im = reshape(cmap(map(:,:,i),:),[size(map,1) size(map,2) 3]);
                    f = fullfile(fullpath, sprintf('aod_monet%03u-%03u-%u.png', ...
                        key.scan_idx, key.point_id, i));
                    imwrite(im, f, 'png')
                end
            end
        end
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % temporal binning
            nbins = 10;
            bin_width = 0.1;  %(s)
            
            disp 'loading traces...'
            [start_time, duration] = fetch1(aodpre.Sync & key, 'signal_start_time', 'signal_duration');
            [X, traceKeys] = fetchn(aodpre.Spikes & key, 'spike_trace');
            ntraces = length(traceKeys);
            X = [X{:}];
            
            % smoothen traces to cutoff frequency
            cutoff = 0.1;
            k = hamming(2*floor(cutoff/duration*size(X,1))+1);
            k = k/sum(k);
            X  = conv2(X,k,'same');

            % inerpolate traces
            caTimes = linspace(0, duration, size(X,1));
            X = @(t) interp1(caTimes, X, t, 'linear', nan);  % traces indexed by time
            
            disp 'loading movies...'
            trials = pro(aodpre.Sync*psy.Trial & key & 'trial_idx between first_trial and last_trial', 'cond_idx', 'flip_times');
            trials = fetch(trials*psy.MovingNoise*psy.MovingNoiseLookup, '*', 'ORDER BY trial_idx');
            sess = fetch(aodpre.Sync*psy.Session & key,'resolution_x','resolution_y','monitor_distance','monitor_size');
            
            % compute physical dimensions
            rect = [sess.resolution_x sess.resolution_y];
            degPerPix = 180/pi*sess.monitor_size*2.54/norm(rect(1:2))/sess.monitor_distance;
            degSize = degPerPix*rect;
            
            disp 'integrating maps...'
            maps = zeros(trials(1).tex_ydim, trials(1).tex_xdim, nbins, ntraces);
            sz = size(maps);
            
            total_frames = 0;
            for trial = trials'
                fprintf('\nTrial %d', trial.trial_idx);
                % reconstruct movie
                movie = (double(trial.cached_movie)-127.5)/127.5;
                trial_frames = length(trial.flip_times);
                movie = fliplr(reshape(movie, [], trial_frames));
                total_frames = total_frames + trial_frames;
                
                % extract relevant trace
                fps = 1/median(diff(trial.flip_times));
                t = trial.flip_times(ceil(nbins*bin_width*fps):end) - start_time;
                snippet = X(t);
                for itrace = 1:ntraces
                    fprintf .
                    update = conv2(movie, snippet(:,itrace)', 'valid')';
                    update = interp1((0:size(update,1)-1)/fps, update, (0:nbins-1)*bin_width);  % resample 
                    update = reshape(update', sz(1), sz(2), nbins);
                    maps(:,:,:,itrace) = maps(:,:,:,itrace) + update;
                end
            end
            fprintf \n
            disp 'inserting...'
            
            maps = maps/total_frames;
            
            for itrace = 1:ntraces
                tuple = dj.struct.join(key, traceKeys(itrace));
                tuple.nbins = nbins;
                tuple.bin_size = bin_width;
                tuple.degrees_x = degSize(1);
                tuple.degrees_y = degSize(2);
                tuple.map = maps(:,:,:,itrace);
                self.insert(tuple)
            end
            
            
        end
    end
    
end