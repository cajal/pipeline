function sync_info = sync(key, photodiode_signal, photodiode_fs)
% given the photodiode signal, returns a structure describing visual
% stimulus trials that were displayed and exact timing synchronization.
% The photodiode signal is assumed to be uniformly sampled in time.

trialTable = vis.Trial;

trials = trialTable & (vis.ScanConditions & key);

if trials.count==0
    [photodiode_flip_indices, photodiode_flip_numbers] = detectFlips(photodiode_signal, photodiode_fs, 30);
    % consider all trials with flip nums within the min and max of detected flip nums
    trials = trialTable & key & ...
        sprintf('last_flip_count between %d and %d', min(photodiode_flip_numbers), max(photodiode_flip_numbers));
    % get all flip times for each trial, also get the number of the last flip in the trial
    [psy_id, last_flip_in_trial, trial_flip_times] = fetchn(trials,...
        'psy_id', 'last_flip_count', 'flip_times', 'ORDER BY trial_idx');
else
    % get all flip times for each trial, also get the number of the last flip in the trial
    [psy_id, last_flip_in_trial, trial_flip_times] = fetchn(trials,...
        'psy_id', 'last_flip_count', 'flip_times', 'ORDER BY trial_idx');
    fps = 1/mean(diff(trial_flip_times{1}));
    [photodiode_flip_indices, photodiode_flip_numbers] = detectFlips(photodiode_signal, photodiode_fs, fps);
end

if any(psy_id ~= psy_id(1))
    warning 'Multiple psy.Sessions per scan: not allowed.'
    same_session = mode(psy_id) == psy_id;
    psy_id = psy_id(same_session);
    last_flip_in_trial = last_flip_in_trial(same_session);
    trial_flip_times = trial_flip_times(same_session);
end
key.psy_id = psy_id(1);
sync_info = key;

% fill in the flip numbers within each trial (counting up to the last flip in the trial)
trial_flip_numbers = arrayfun(@(last_flip, flip_times) ...
    last_flip+(1-length(flip_times{1}):0), last_flip_in_trial, trial_flip_times, 'uni', false);
trial_flip_numbers = cat(2, trial_flip_numbers{:});
trial_flip_times = [trial_flip_times{:}];
assert(length(trial_flip_times)==length(trial_flip_numbers));

% Select only the matched flips
ix = ismember(photodiode_flip_numbers, trial_flip_numbers);
assert(sum(ix)>100, 'Insufficient matched flips (%d)', sum(ix))
photodiode_flip_indices = photodiode_flip_indices(ix);
photodiode_flip_numbers = photodiode_flip_numbers(ix);
trial_flip_times = trial_flip_times(ismember(trial_flip_numbers, photodiode_flip_numbers));

% regress the photodiode_signal indices against the stimulus times to get the
% photodiode_signal signal time on stimulus clock. Assumes uninterrupted uniform sampling of photodiode_signal!!!
photodiode_flip_times = photodiode_flip_indices/photodiode_fs;
b = robustfit(photodiode_flip_times, trial_flip_times-trial_flip_times(1));
sync_info.signal_start_time = b(1) + trial_flip_times(1);
sync_info.signal_duration = length(photodiode_signal)/photodiode_fs*b(2);
time_discrepancy =  (b(1) + photodiode_flip_times*b(2)) -  (trial_flip_times(:)-trial_flip_times(1));
assert((quantile(abs(time_discrepancy),0.999)) < 0.1, ...
    'Incorrectly detected flips. Time discrepancy = %f s', max(abs(time_discrepancy)))

% find first and last trials overlapping signal
trials = fetch(trialTable & key, 'trial_idx', 'flip_times', 'ORDER BY trial_idx');
i=1;
while trials(i).flip_times(end) < sync_info.signal_start_time
    i=i+1;
end
sync_info.first_trial = trials(i).trial_idx;
i=length(trials);
while trials(i).flip_times(1) > sync_info.signal_start_time + sync_info.signal_duration
    i=i-1;
end
sync_info.last_trial = trials(i).trial_idx;

end

function [photodiode_flip_indices, photodiode_flip_numbers] = detectFlips(photodiode_signal, photodiode_fs, fps)

    % detect flips in the recorded photodiode_signal signal
    [photodiode_flip_indices, photodiode_flip_numbers] = ...
        stims.analysis.whichFlips(photodiode_signal, photodiode_fs, fps);

    % remove duplicated numbers due to terminated programs
    ix = ~isnan(photodiode_flip_numbers);
    photodiode_flip_indices = photodiode_flip_indices(ix);
    photodiode_flip_numbers = photodiode_flip_numbers(ix);
    ix = find(diff(photodiode_flip_numbers)<0,1, 'last');
    if ~isempty(ix)
        photodiode_flip_indices = photodiode_flip_indices(ix+1:end);
        photodiode_flip_numbers = photodiode_flip_numbers(ix+1:end);
    end

    assert(~isempty(photodiode_flip_numbers), 'no flips detected in photodiode channel')
end