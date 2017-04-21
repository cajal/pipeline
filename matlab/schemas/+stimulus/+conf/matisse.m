function matisse
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions

params = struct(...
    'noise_seed', 1:3,    ...   from vis.BaseNoise128
    'pre_blank_period', 0.2, ...  (seconds)
    'duration', 0.2, ... (seconds)
    'pattern_width', 64, ...    cannot exceed 128
    'pattern_aspect', 16/9, ...  the aspect ratio of the pattern -- should match the screen
    'ori', 0:15:179, ... degrees. 0=horizontal, then clockwise
    'outer_ori_delta', -90:45:89, ...  degrees. Differerence of outer ori from inner.
    'ori_coherence', [1.5 4], ...   1=unoriented noise. pi/ori_coherence = bandwidth of orientations.
    'aperture_x', -0.2, ... x position of the aperture in units of pattern widths: 0=center, 0.5=right edge
    'aperture_y', -0.1, ... y position of the aperture in units of pattern widths: 0=center, 0.5=right edge
    'aperture_r', 0.15, ... aperture radius expressed in units pattern widths
    'aperture_transition', 0.05, ...  aperture transition width
    'annulus_alpha', 0, ... aperture annulus alpha
    'inner_contrast', 1, ...  pattern contrast in inner region
    'outer_contrast', 1 ...  pattern contrast in outer region
);

assert(isscalar(params))
params = stimulus.utils.factorize(params);

% generate conditions
hashes = control.makeConditions(stimulus.Matisse, params);

% push trials
nblocks = 3;
for i=1:nblocks
    control.pushTrials(hashes(randperm(numel(hashes))))
end
end