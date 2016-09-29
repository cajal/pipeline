%{
monet.MonetRF (computed) # receptive fields from the monet stimulus
-> preprocess.Sync
-> preprocess.Spikes
-----
nbins              : smallint                      # temporal bins
bin_size           : float                         # (ms) temporal bin size
degrees_x          : float                         # degrees along x
degrees_y          : float                         # degrees along y
stim_duration      : float                         # (s) total stimulus duration
%}

classdef MonetRF < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = preprocess.Spikes * preprocess.Sync & vis.Monet
    end
       
    methods(Access=protected)
        
        function makeTuples(self, key)
            % temporal binning
            nbins = 6;
            bin_width = 0.1;  %(s)
            
            disp 'loading traces...'
            [X, caTimes, traceKeys] = pipetools.getAdjustedSpikes(key);
%             lenDif = length(caTimes)-size(X,1);
%             if lenDif<nslices && lenDif>0 % aborted scans can give rise to unequal ca traces!
%                 warning('Unequal vectors, equilizing caTimes...')
%                 caTimes = caTimes(1:end-lenDif);
%             elseif abs(lenDif)>=nslices
%                 error('Ca traces & stimulus vector significantly different!');
%             end
            X = @(t) interp1(caTimes-caTimes(1), X, t, 'linear', nan);  % traces indexed by time
            ntraces = length(traceKeys);
            
            disp 'loading movies...'
            trials = pro(preprocess.Sync * vis.Trial * vis.Condition & ...
                'trial_idx between first_trial and last_trial' & ...
                vis.Monet & key, 'cond_idx', 'flip_times');
            trials = fetch(trials, '*', 'ORDER BY trial_idx');
            sess = fetch(preprocess.Sync*vis.Session & key,'resolution_x','resolution_y','monitor_distance','monitor_size');
            
            % compute physical dimensions
            rect = [sess.resolution_x sess.resolution_y];
            degPerPix = 180/pi*sess.monitor_size*2.54/norm(rect(1:2))/sess.monitor_distance;
            degSize = degPerPix*rect;
            
            disp 'integrating maps...'
            keys = fetch(vis.Monet * vis.MonetLookup & key);
            map = fetch1(vis.Monet * vis.MonetLookup & keys(1),'cached_movie');
            maps = zeros(size(map,1), size(map,2), nbins, ntraces);
            sz = size(maps);
            
            total_frames = 0;
            for trial = trials'
                fprintf('\nTrial %d', trial.trial_idx);
                % reconstruct movie
                cached_movie = fetch1(vis.Monet * vis.MonetLookup & trial,'cached_movie');
                movie = (double(cached_movie)-127.5)/127.5;
                trial_frames = length(trial.flip_times);
                movie = fliplr(reshape(movie, [], trial_frames));
                total_frames = total_frames + trial_frames;
                
                % extract relevant trace
                fps = 1/median(diff(trial.flip_times));
                t = trial.flip_times(ceil(nbins*bin_width*fps):end) - caTimes(1);
                snippet = X(t);
                for itrace = 1:ntraces
                    fprintf .
                    update = conv2(movie, snippet(:,itrace)', 'valid')';
                    update = interp1((0:size(update,1)-1)/fps, update, (0:nbins-1)*bin_width);  % resample 
                    update = reshape(update', sz(1), sz(2), nbins);
                    if any(isnan(update(:)));fprintf('nan values for trial # %d, skipping...',trial.trial_idx);break;end
                    maps(:,:,:,itrace) = maps(:,:,:,itrace) + update;
                end
            end
            
            fprintf \n
            disp 'inserting...'
            
            maps = maps/total_frames;

            tuple = key;
            tuple.nbins = nbins;
            tuple.bin_size = bin_width;
            tuple.degrees_x = degSize(1);
            tuple.degrees_y = degSize(2);
            tuple.stim_duration = total_frames/fps;
            self.insert(tuple)
            
            for itrace = 1:ntraces
                tuple = dj.struct.join(key, rmfield(traceKeys(itrace),'slice'));
                tuple.map = single(maps(:,:,:,itrace));
                makeTuples(tuning.MonetRFMap,tuple)
            end
            
        end
    end
    
end