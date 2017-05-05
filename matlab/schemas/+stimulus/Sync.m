%{
# synchronization of stimulus to scan
-> experiment.Scan
---
signal_start_time           : double                        # (s) signal start time on stimulus clock
signal_duration             : double                        # (s) signal duration on stimulus time
frame_times=null            : longblob                      # times of frames and slices on stimulus clock
sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
%}

classdef Sync < dj.Imported
    
    
    properties
        keySource = experiment.Scan & preprocess.Prepare
    end
    
    methods(Static)
        function migrate
            % migrate from the legacy schema vis
            % This is incremental: can be called multiple times
            missing = preprocess.Sync - proj(stimulus.Sync);
            ignore_extra = dj.set('ignore_extra_insert_fields', true);
            insert(stimulus.Sync, missing.fetch('*'))
            dj.set('ignore_extra_insert_fields', ignore_extra);
        end
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            % read photodiode signal
            dat = preprocess.readHD5(key);
            packetLen = 2000;
            if isfield(dat,'analogPacketLen')
                packetLen = dat.analogPacketLen;
            end
            datT = pipetools.ts2sec(dat.ts, packetLen);
            photodiode_fs = 1/median(diff(datT));
            photodiode_signal = dat.syncPd;
            fps = 60;   % does not need to be exact
            
            % synchronize to stimulus
            tuple =  sync(key, photodiode_signal,...
                photodiode_fs, fps, stimulus.Trial & key);
            
            self.insert(key)
        end
    end
    
end


function sync_info = sync(key, photodiode_signal, photodiode_fs, fps, trials)
% given the photodiode signal, returns a structure describing visual
% stimulus trials that were displayed and exact timing synchronization.
% The photodiode signal is assumed to be uniformly sampled in time.

% detect flips in the recorded photodiode_signal signal
[photodiode_flip_indices, photodiode_flip_numbers] = ...
    whichFlips(photodiode_signal, photodiode_fs, fps);

% remove duplicated numbers due to terminated programs
ix = ~isnan(photodiode_flip_numbers);
photodiode_flip_indices = photodiode_flip_indices(ix);
photodiode_flip_numbers = photodiode_flip_numbers(ix);

% if flips are ouf of order, reset to the start of the most recent count
ix = find(photodiode_flip_numbers(2:end) <= photodiode_flip_numbers(1:end-1), 1, 'last');
if ~isempty(ix)
    photodiode_flip_indices = photodiode_flip_indices(ix+1:end);
    photodiode_flip_numbers = photodiode_flip_numbers(ix+1:end);
end

assert(~isempty(photodiode_flip_numbers), 'no flips detected in photodiode channel')

% get the flip times and numbers from the trials
[last_flip_in_trial, trial_flip_times] = fetchn(trials, ...
    'last_flip', 'flip_times', 'ORDER BY trial_idx');
trial_flip_numbers = arrayfun(@(last_flip, flip_times) ...
    last_flip+(1-length(flip_times{1}):0), last_flip_in_trial, trial_flip_times, 'uni', false);
trial_flip_numbers = [trial_flip_numbers{:}];
trial_flip_times = [trial_flip_times{:}];
assert(length(trial_flip_times)==length(trial_flip_numbers));

assert(all(ismember(photodiode_flip_numbers, trial_flip_numbers)), ...
    'some photodiode numbers come from a different scan')

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
assert((quantile(abs(time_discrepancy),0.999)) < 0.034, ...
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



function [flipIdx, flipNums] = whichFlips(x, fs, fps)
% given a photodiode signal x with sampling rate fs, decode photodiode
% flips numbers.  The monitor frame rate is fps.
%
% returns:
% flipIdx: the indices of the detected flips in x and their encoded
% flipNums: the sequential numbers of the detected indices

[flipIdx, flipAmps] = getFlips(x, fs, fps);
flipNums = flipAmpsToNums(flipAmps);
end



function [flipIdx, flipAmps] = getFlips(x, fs, frameRate)
% INPUTS:
%   x - photodiode signal
%   fs - (Hz) sampling frequency
%   frameRate (Hz) monitor frame rate

T = fs/frameRate*2;  % period of oscillation measured in samples
% filter flips
n = floor(T/4);  % should be T/2 or smaller
k = hamming(n);
k = [k;0;-k]/sum(k);
x = fftfilt(k,[double(x);zeros(n,1)]);
x = x(n+1:end);
x([1:n end+(-n+1:0)])=0;  % remove edge artifacts

% select flips
flipIdx = ne7.dsp.spaced_max(abs(x),0.22*T);
thresh = 0.15*quantile( abs(x(flipIdx)),0.999);
flipIdx = flipIdx(abs(x(flipIdx))>thresh)';
flipAmps = x(flipIdx);
end



function flipNums = flipAmpsToNums(flipAmps)
% given a sequence of flip amplitudes with encoded numbers,
% assign cardinal numbers to as many flips as possible.

flipNums = nan(size(flipAmps));

% find threshold for positive flips (assumed stable)
ix = find(flipAmps>0);
thresh = (quantile(flipAmps(ix),0.1) + quantile(flipAmps(ix),0.9))/2;

frame = 16; % 16 positive flips
nFrames = 5;  % must be odd. 3 or 5 are most reasonable
iFlip = 1;
quitFlip = length(flipAmps)-frame*nFrames*2-2;
while iFlip < quitFlip
    amps = flipAmps(iFlip+(0:frame*nFrames-1)*2);
    if all(amps>0) % only consider positive flips
        bits = amps < thresh;  % big flips are for zeros
        nums = bin2dec(char(fliplr(reshape(bits, [frame nFrames])')+48));
        if all(diff(nums)==1)  % found sequential numbers
            %fill out the numbers of the flips in the middle frame
            ix = iFlip + floor(nFrames/2)*frame*2 + (0:frame*2-1);
            nums = nums((nFrames+1)/2)*frame*2 + (1:frame*2);
            flipNums(ix) = nums;
            iFlip = iFlip + frame*2-1;
        end
    end
    iFlip = iFlip+1;
end
end