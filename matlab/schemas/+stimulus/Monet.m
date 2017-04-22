%{
# pink noise with periods of motion and orientation
-> stimulus.Condition
---
moving_noise_version        : smallint                      # algorithm version; increment when code changes
rng_seed                    : double                        # random number generate seed
fps                         : decimal(5,2)                  # display refresh rate
tex_ydim                    : smallint                      # (pixels) texture dimension
tex_xdim                    : smallint                      # (pixels) texture dimension
spatial_freq_half           : float                         # (cy/deg) spatial frequency modulated to 50 percent
spatial_freq_stop           : float                         # (cy/deg), spatial lowpass cutoff
temp_bandwidth              : float                         # (Hz) temporal bandwidth of the stimulus
ori_on_secs                 : float                         # seconds of movement and orientation
ori_off_secs                : float                         # seconds without movement
n_dirs                      : smallint                      # number of directions
ori_bands                   : tinyint                       # orientation width expressed in units of 2*pi/n_dirs
ori_modulation              : float                         # mixin-coefficient of orientation biased noise
speed                       : float                         # (degrees/s)
x_degrees                   : float                         # degrees across x if screen were wrapped at shortest distance
y_degrees                   : float                         # degrees across y if screen were wrapped at shortest distance
directions                  : blob                          # (degrees) directions in periods of motion
onsets                      : blob                          # (s) the times of onsets of moving periods
movie                       : longblob                      # actual movie
%}


classdef Monet < dj.Manual & stimulus.core.Visual
    % Legacy Monet stimulus used for migrating from +psy but not for
    % running new stimuli. The code for generating the stimulus is in
    % psy.MonetLookup
    
    properties(Constant)
        version = 'legacy from vis.Monet * vis.MonetLookup'
    end
    
    methods(Static)
        function migrate
            % migrate data from the legacy schema +vis
            control = stimulus.getControl;            
            
            % migrate monet conditions
            monet = vis.MonetLookup*vis.Monet & (vis.Trial*preprocess.Sync & 'trial_idx between first_trial and last_trial' & 'animal_id>0');
            condKeys = monet.fetch';
            count = 0;
            for key = condKeys
                count = count + 1;
                fprintf('[%d/%d]', count, length(condKeys)) 
                cond = fetch(vis.MonetLookup*vis.Monet & key, '*');
                assert(cond.frame_downsample == 1)
                params = cond.params;
                cond.movie = cond.cached_movie;
                cond = rmfield(cond, {'animal_id', 'psy_id', 'cond_idx', ...
                    'moving_noise_paramhash', 'params', 'frame_downsample', ...
                    'cached_movie', 'moving_noise_lookup_ts', 'luminance', 'contrast'});
                cond.fps = params{3};
                cond.directions = params{4}.direction;
                cond.onsets = params{4}.onsets;
                cond.x_degrees = params{2}(1);
                cond.y_degrees = params{2}(2);                
                hash = control.makeConditions(stimulus.Monet, cond);
                
                % copy all trials that used this condition
                for tuple = fetch(vis.Trial*preprocess.Sync & 'trial_idx between first_trial and last_trial' & key,...
                        'last_flip_count->last_flip', 'trial_ts', 'flip_times')'
                    insert(stimulus.Trial, setfield(rmfield(tuple, {'psy_id'}), 'condition_hash', hash{1}))
                end
            end            
        end
    end
    
    methods
        function showTrial(cond)
            error 'This is a legacy stimulus for analysis only.  The showTrial code is visual-stimulus/+stims/Monet.m'
        end
    end
end