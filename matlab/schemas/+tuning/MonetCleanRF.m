%{
tuning.MonetCleanRF (computed) # RF maps with common components removed
-> tuning.MonetRFMap
---
clean_map                   : longblob                      # 
%}


classdef MonetCleanRF < dj.Relvar & dj.AutoPopulate

	properties
		popRel = tuning.MonetRFMap  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		    disp 'loading RFs and traces'
            [maps, bin_width, keys] = fetchn(tuning.MonetRF * tuning.MonetRFMap & key, 'map', 'bin_size');
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
            [Traces, caTimes] = pipetools.getAdjustedSpikes(key);
            X = @(t) interp1(caTimes-caTimes(1), Traces, t, 'linear', nan);  % traces indexed by time
            trials = pro(preprocess.Sync*vis.Trial & key & 'trial_idx between first_trial and last_trial', 'cond_idx', 'flip_times');
            trials = fetch(trials*vis.Monet*vis.MonetLookup, 'flip_times', 'ORDER BY trial_idx');
            
            disp 'normalizing RFs...'            
            total_frames = 0;
            vars = 0;
            for trial = trials'
                total_frames = total_frames + length(trial.flip_times);
                fps = 1/median(diff(trial.flip_times));
                t = trial.flip_times(ceil(nbins*bin_width*fps):end) - caTimes(1);
                s = sum(X(t).^2);
                if any(isnan(s(:)));fprintf('nan values for trial # %d, skipping...',trial.trial_idx);break;end
                vars = vars + s;
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