function grating
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions

params = struct(...
    'monitor_distance_ratio', 0.58, ...  the distance to the screen measured in screen diagonals
    'direction', [0 90], ...  0-360 degrees
    'spatial_freq', 0.08, ... cycles/degree
    'temp_freq', 4, ...  Hz
    'pre_blank', 3, ... (s) blank period preceding trials
    'luminance', 5, ... cd/m^2 mean
    'contrast', 0.95, ... Michelson contrast 0-1
    'aperture_radius', 0.15, ... in units of half-diagonal, 0=no aperture
    'aperture_x', [-0.4 0.2], ...  aperture x coordinate, in units of half-diagonal, 0 = center
    'aperture_y', [-0.36 0.32], ... aperture y coordinate, in units of half-diagonal, 0 = center
    'shape', 'sin', ... sin or square, etc.
    'init_phase', 0, ... 0..1
    'trial_duration', 2, ...  s, does not include pre_blank duration
    'phase2_fraction', 0, ... fraction of trial spent in phase 2
    'phase2_temp_freq', 0, ... (Hz)
    'second_photodiode', 0, ... 1=paint a photodiode patch in the upper right corner
    'second_photodiode_time', 0 ... time delay of the second photodiode relative to the stimulus onset
);

assert(isscalar(params))
params = stimulus.utils.factorize(params);

% generate conditions
hashes = control.makeConditions(stimulus.Grating, params);

% push trials
nblocks = 3;
for i=1:nblocks
    control.pushTrials(hashes(randperm(numel(hashes))))
end
end
