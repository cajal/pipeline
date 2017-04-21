function grate
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions

% specify conditions as factorial design
params = struct(...
    'monitor_distance_ratio', 0.58, ...  the distance to the screen measured in screen diagonals
    'pre_blank', 0.5, ... (s) blank period preceding trials
    'duration', 0.5, ...  s, does not include pre_blank duration
    'direction', 0:15:359, ...  0-360 degrees
    'spatial_freq', 0.08, ... cycles/degree
    'temp_freq', 3, ...  Hz
    'aperture_radius', 0.15, ... in units of half-diagonal, 0=no aperture
    'aperture_x', [-0.40 0.40], ...  aperture x coordinate, in units of half-diagonal, 0 = center
    'aperture_y', [-0.32 0.32], ... aperture y coordinate, in units of half-diagonal, 0 = center
    'init_phase', 0 ... 0..1
);

% generate conditions
assert(isscalar(params))
params = stimulus.utils.factorize(params);

% save conditions
hashes = control.makeConditions(stimulus.Grate, params);

% queue trials 
nblocks = 3;
fprintf('Total time: %g s\n', nblocks*(sum([params.duration]) + sum([params.pre_blank])))
for i=1:nblocks
    control.pushTrials(hashes(randperm(numel(hashes))))
end
end
