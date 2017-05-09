function matisse2

control = stimulus.getControl;
control.clearAll   % clear trial queue and cached conditions.

cond.fps = 60;
cond.pre_blank_period = 0.5;
cond.pattern_width = 64;
cond.duration = 1;
cond.pattern_aspect = 1.7;
cond.ori = 0:45:359;
cond.outer_ori_delta = -30:20:30;
cond.coherence = 1.5;
cond.aperture_x = -0.2:0.1:0.2;
cond.aperture_y = -0.1:0.1:0.1;
cond.aperture_r = 0.2;
cond.aperture_transition = 0.1;
cond.annulus_alpha = 0.0;
cond.outer_contrast = 1;
cond.inner_contrast = 1;
cond.outer_speed = 0.2;
cond.inner_speed = 0.2;

% assert(isscalar(cond))
params = stimulus.utils.factorize(cond);
seed = num2cell(1:numel(params));
[params.noise_seed] = deal(seed{:});

fprintf('Total duration: %4.2f s\n', (sum([params.duration]) + sum([params.pre_blank_period])))

% generate conditions
hashes = control.makeConditions(stimulus.Matisse2, params);

% queue trials
control.pushTrials(hashes(randperm(numel(hashes))))
end