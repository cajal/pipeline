function Grate
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions

params = struct(...
    'monitor_distance_ratio', 0.58, ...  the distance to the screen measured in screen diagonals
    'pre_blank', 0.5, ... (s) blank period preceding trials
    'duration', 0.5, ...  s, does not include pre_blank duration
    'direction', 0:15:359, ...  0-360 degrees
    'spatial_freq', 0.08, ... cycles/degree
    'temp_freq', 3, ...  Hz
    'aperture_radius', 0.15, ... in units of half-diagonal, 0=no aperture
    'aperture_x', [-0.4 0.2], ...  aperture x coordinate, in units of half-diagonal, 0 = center
    'aperture_y', [-0.36 0.32], ... aperture y coordinate, in units of half-diagonal, 0 = center
    'init_phase', 0 ... 0..1
);

assert(isscalar(params))
params = stimulus.utils.factorize(params);

fprintf('Total time per block: %g s\n', sum([params.duration]) + sum([params.pre_blank]))

% generate conditions
hashes = control.makeConditions(stimulus.Grate, params);

% push trials
nblocks = 3;
for i=1:nblocks
    control.pushTrials(hashes(randperm(numel(hashes))))
end
end