%{
a stand-alone script for Monet stimulus processing of galvo traces
%}

%% prepare
matched_trials = preprocess.Sync * vis.Trial & 'trial_idx between first_trial and last_trial';

%% all galvo scans for which there is a monet stimulus
monet_keys = fetch(preprocess.Spikes*preprocess.SpikeMethod*preprocess.PrepareGalvo*preprocess.MethodGalvo & ...
    (matched_trials & vis.Monet) & 'spike_method_name="stm"' & 'segmentation="nmf"');

%% select one of the datasets
key = monet_keys(1);

%% Alternatively, start with a known dataset
key = struct(...
    'animal_id', 9161, ...
    'session', 1, ...
    'scan_idx', 11, ...
    'extract_method', 2, ...
    'spike_method', 3);

%% get spike traces
[traces, slice] = fetchn(preprocess.SpikesRateTrace*preprocess.ExtractRawGalvoROI & key, 'rate_trace', 'slice');
nslices = fetch1(preprocess.PrepareGalvo & key, 'nslices');
traces = double([traces{:}]);  % to 2d array

%% get time
time = fetch1(preprocess.Sync & key, 'frame_times');
time_between_slices = mean(diff(time));
time = time(1:nslices:end);   % downsample to frame rate. To get precise time of each trace, do time + (slice-1)*time_between_slices

%% get movies one-by-one to save memory
for movie_key = fetch(matched_trials & key)'
    movie_info = fetch(vis.Trial * vis.Monet * vis.MonetLookup & movie_key, '*');
    
    disp 'Movie info'
    disp(movie_info)
    
    disp 'Drifting periods'
    disp(movie_info.params{4})
    
    ...... do your processing here using time, traces, and movie_info .....
        
end