function Matisse2

control = stimulus.getControl;
control.clearAll   % clear trial queue and cached conditions.

cond.fps = 60;
cond.pre_blank_period = 0.1;
cond.noise_seed = 100;
cond.pattern_width = 64;
cond.pattern_upscale = 3;
cond.duration = 1;
cond.pattern_aspect = 1.7;
cond.ori = 0:90:359;
cond.outer_ori_delta = 90;
cond.coherence = [1 1.5 2.5];
cond.aperture_x = -0.10;
cond.aperture_y = +0.05;
cond.aperture_r = 0.2;
cond.aperture_transition = 0.1;
cond.annulus_alpha = 0;
cond.outer_contrast = [0.1 0.2 1];
cond.inner_contrast = [0.1 0.2 1];
cond.outer_speed = 0.2;
cond.inner_speed = 0.2;

% assert(isscalar(cond))
params = stimulus.utils.factorize(cond);
nblocks = 2;
fprintf('Total duration: %4.2f s\n', nblocks*(sum([params.duration]) + sum([params.pre_blank_period])))

% generate conditions
hashes = control.makeConditions(stimulus.Matisse2, params);

% push trials
for i=1:nblocks
    control.pushTrials(hashes(randperm(numel(hashes))))
end
end