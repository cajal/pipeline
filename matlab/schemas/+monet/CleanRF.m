%{
monet.CleanRF (computed) # cleaned up and normalized receptive fields
-> monet.RF
-----
clean_map : longblob   # cleaned up receptive field map
%}

classdef CleanRF < dj.Relvar & dj.AutoPopulate

	properties
        popRel = pre.ExtractSpikes & monet.RF
	end

	methods(Access=protected)

		function makeTuples(self, key)
            disp 'loading RFs and traces'
            [maps, X, bin_width, keys] = fetchn(monet.RF*pre.Spikes & key, 'map', 'spike_trace', 'bin_size');
            assert(all(bin_width==bin_width(1)))
            bin_width = bin_width(1);
            nmaps = length(maps);
            maps = cat(4,maps{:});
            sz = size(maps);
            nbins = sz(3);
            if nmaps > 5
                % subtract the first principal component if it has the same
                % signs in the cell space.
                maps = double(reshape(maps, [], nmaps));            
                disp 'removing effects of population spikes...'
                [U,D,V] = svds(bsxfun(@minus, maps, mean(maps)), 1); 
                if all(sign(V)==sign(V(1)))
                    maps = reshape(maps-U*D*V', sz);
                    disp 'removed 1 component'
                end
            end
            maps = reshape(maps, sz);
            
            disp 'loading trials...'
            caTimes = fetch1(rf.Sync & key, 'frame_times'); 
            nslices = fetch1(pre.ScanInfo & key, 'nslices');
            caTimes = caTimes(key.slice:nslices:end);
            X = [X{:}];
            lenDif = length(caTimes)-size(X,1);
            if lenDif<nslices && lenDif>0 % aborted scans can give rise to unequal ca traces!
                warning('Unequal vectors, equilizing caTimes...')
                caTimes = caTimes(1:end-lenDif);
            else
                error('CaTrace & time vectors have significantly different sizes!');
            end
            X = @(t) interp1(caTimes-caTimes(1), X, t, 'linear', nan);  % traces indexed by time
            trials = pro(rf.Sync*psy.Trial & key & 'trial_idx between first_trial and last_trial', 'cond_idx', 'flip_times');
            trials = fetch(trials*psy.MovingNoise*psy.MovingNoiseLookup, 'flip_times', 'ORDER BY trial_idx');
            
            disp 'normalizing RFs...'            
            total_frames = 0;
            vars = 0;
            for trial = trials'
                total_frames = total_frames + length(trial.flip_times);
                fps = 1/median(diff(trial.flip_times));
                t = trial.flip_times(ceil(nbins*bin_width*fps):end) - caTimes(1);
                vars = vars + sum(X(t).^2);
            end
            vars = vars/total_frames;
            nsigmas = norminv(1e-5/2);  % inverse of movie contrast expressed in standard deviations.  This is derived from the fact that the movie was clipped to 1-1e-5 qauntile
            maps = nsigmas*bsxfun(@rdivide, maps, reshape(sqrt(vars), 1, 1, 1, nmaps));
            
            for imap = 1:nmaps
                self.insert(setfield(keys(imap), 'clean_map', single(maps(:,:,:,imap)))) %#ok<SFLD>
            end
		end
	end

end