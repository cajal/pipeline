%{
a stand-alone script for MadMax stimulus processing of galvo traces
%}

%% prepare
matched_trials = preprocess.Sync * vis.Trial * vis.Condition & 'trial_idx between first_trial and last_trial';

%% all galvo scans for which there is a monet stimulus
madmax_keys = fetch(preprocess.Spikes*preprocess.SpikeMethod*preprocess.MethodGalvo & ...
    (matched_trials & (vis.MovieClipCond | vis.MovieStillCond | vis.MovieSeqCond)) & 'spike_method_name="nmf"' & 'segmentation="nmf"');

%% select one of the datasets
key = madmax_keys(1);

%% Alternatively, start with a known dataset
key = struct(...
    'animal_id', 9036, ...
    'session', 1, ...
    'scan_idx', 3, ...
    'extract_method', 2, ...
    'spike_method', 5);

%% get spike traces
[traces, slice] = fetchn(preprocess.SpikesRateTrace*preprocess.ExtractRawGalvoROI & key, 'rate_trace', 'slice');
nslices = fetch1(preprocess.PrepareGalvo & key, 'nslices');
traces = double([traces{:}]);  % to 2d array

%% get time
time = fetch1(preprocess.Sync & key, 'frame_times');
time_between_slices = mean(diff(time));
time = time(1:nslices:end);   % downsample to frame rate. To get precise time of each trace, do time + (slice-1)*time_between_slices

%% get movies one-by-one to save memory
temp_movie_file = './temp.mov';
show_frames = false;
for stim_key = fetch(matched_trials & key, 'ORDER BY trial_idx')'   % in chronological order
    
    switch true
        case exists(vis.MovieClipCond & stim_key)
            % process movie clip
            info = fetch(vis.Trial * vis.MovieClipCond * vis.MovieClip & stim_key, '*');
            assert(length(info)==1)
            
            % write the compressed movie into temp_movie_file
            fid = fopen(temp_movie_file, 'w');
            fwrite(fid, info.clip, 'int8')
            fclose(fid);
            
            % read the movie from temp_movie_file
            if show_frames
                v = VideoReader(temp_movie_file);
                while hasFrame(v)
                    imshow(readFrame(v))
                    drawnow
                end
            end
            disp(info)
            
            ....... do your processing here using traces, time, info, and temp_movie_file  .......
                
        
        
        case exists(vis.MovieStillCond & stim_key)
            % process still images
            info = fetch(vis.Trial * vis.MovieStillCond * vis.MovieStill & stim_key, '*');
            disp(info)
            
            ....... do your processing here using traces, time, and info  .......
                
        
        case exists(vis.MovieSeqCond & stim_key)
            % process sequences of still images
            info = fetch(vis.Trial * vis.MovieSeqCond & stim_key, '*');
            frames_shown = fetchn(vis.MovieStill & key & struct('still_id', num2cell(info.movie_still_ids)), 'still_frame');
  
            ....... do your processing here using traces, info, and frames_shown .......
                
        otherwise 
            error 'another kind of stimulus was used for this trial'
                
    end
end
