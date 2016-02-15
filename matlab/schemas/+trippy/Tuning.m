%{
trippy.Tuning (computed) # local directional tuning maps
-> rf.Sync
-> pre.Spikes
-----
magnitude           :longblob     #   map of tuning magnitude
pref_direction      :longblob     #   map of preferred directions
%}

classdef Tuning < dj.Relvar & dj.AutoPopulate

    properties
        popRel  = pre.ExtractSpikes*(rf.Sync & psy.Trippy)
    end

    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % temporal binning
            nbins = 6;
            bin_width = 0.1;  %(s)
            dfactor = 4;  % oversampling
            
            disp 'loading traces...'
            caTimes = fetch1(rf.Sync & key, 'frame_times');
            dt = median(diff(caTimes));
            [X, traceKeys] = fetchn(pre.Spikes & key, 'spike_trace');
            X = [X{:}];
            X = @(t) interp1(caTimes-caTimes(1), X, t, 'linear', nan);  % traces indexed by time
            ntraces = length(traceKeys);
            
            disp 'loading movies...'
            trials = pro(rf.Sync*psy.Trial & key & 'trial_idx between first_trial and last_trial', 'cond_idx', 'flip_times');
            trials = fetch(trials*psy.Trippy, '*', 'ORDER BY trial_idx');
            sess = fetch(rf.Sync*psy.Session & key,'resolution_x','resolution_y','monitor_distance','monitor_size');
            fps = round(1./median(diff(trials(1).flip_times)));
            
            % compute physical dimensions
            rect = [sess.resolution_x sess.resolution_y];
            degPerPix = 180/pi*sess.monitor_size*2.54/norm(rect(1:2))/sess.monitor_distance;
            degSize = degPerPix*rect;
            
            disp 'integrating maps...'
            maps = zeros(trials(1).tex_ydim, trials(1).tex_xdim, nbins, ntraces);
            sz = size(maps);
            
            total_frames = 0;
            ndirs = 24;
            central_spatial_freq = 0.1;
            for trial = trials'
                fprintf('\nTrial %d', trial.trial_idx);
                % reconstruct movie
                phase = psy.Trippy.interp_time(trial.packed_phase_movie, trial, fps/trial.frame_downsample);
                trial_frames = size(phase,1);
                assert(trial_frames == length(trial.flip_times), 'frame number mismatch')
                phase = psy.Trippy.interp_space(phase, trial);
                assert(size(phase,1)==sz(1) && size(phase,2)==sz(2))
                movie = cos(2*pi*phase);
                [gx, gy] = gradient(phase, degPerPix);
                direction = max(1, min(ndirs, ceil(mod(atan2(gy,gx), 2*pi)/(2*pi)*ndirs)));
                spatial_freq = sqrt(gx.^2 + gy.^2);
                temp_freq = (phase(:,:,[2:end end]) - phase(:,:,[1 1:end-1]))/dt/2;                
                spatial_freq_window = exp(-(log(spatial_freq)-log(central_spatial_freq)).^2/2);
                total_frames = total_frames + trial_frames;
                
                
                % extract relevant trace
                t = trial.flip_times(ceil(nbins*bin_width/dt):end) - caTimes(1);
                snippet = X(t);
                for itrace = 1:ntraces
                    fprintf .
                    update = conv2(movie, snippet(:,itrace)', 'valid')';
                    update = interp1((0:size(update,1)-1)*dt, update, (0:nbins*dfactor-1)*bin_width/dfactor);
                    update = downsample(conv2(update, ones(dfactor, 1)/dfactor, 'valid'), dfactor)';
                    update = reshape(update, sz(1), sz(2), nbins);
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