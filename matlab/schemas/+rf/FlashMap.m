%{
rf.FlashMap (imported) #  site-wide fast retinotopy
-> rf.Sync
-----
response_map :longblob  # map of response amplitudes on the screen
x_map :longblob  # map of screen locations in brain pixels
y_map :longblob  # map of screen locations in brain pixels
%}

classdef FlashMap < dj.Relvar & dj.AutoPopulate

	properties
		popRel = rf.Sync & psy.FlashingBar
	end

	methods(Access=protected)

		function makeTuples(self, key)
            frameTimes = fetch1(rf.Sync & key, 'frame_times');
            trialRel = rf.Sync*psy.Trial*psy.FlashingBar & key & ...
                'trial_idx between first_trial and last_trial';
            s = trialRel.fetch('flip_times', 'orientation', 'offset');
            [fliptimes, orientation, offset] = dj.struct.tabulate(s, ...
                'flip_times', 'orientation','offset');
            reader = pre.getReader(key);
            sz = size(reader);
            assert(sz(5)*sz(4)==length(frameTimes))
            siteTrace = arrayfun(@(i) squeeze(mean(mean(reader(:,:,1,:,i)))),  1:sz(5), 'uni', false);
            siteTrace = cat(1,siteTrace{:});
            computeResponse = @(flips) mean(interp1(frameTimes, siteTrace, flips, 'nearest'));
            responses = cellfun(computeResponse, fliptimes);
            
            responses = mean(responses, 3);  % average trials
            
            figure
            subplot 121
            plot(offset,responses')
            subplot 122
            imagesc(offset, offset, responses(1,:)'*responses(2,:)),
            axis image
            drawnow
            
			%self.insert(key)
		end
    end
end

