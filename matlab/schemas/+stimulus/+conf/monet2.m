function monet2

control = stimulus.getControl;
control.clearAll   % clear trial queue and cached conditions.

cond.fps = 60;
cond.duration = 30;
cond.rng_seed = 1:60;
cond.pattern_width = 72;
cond.pattern_aspect = 1.7;
cond.ori_coherence = 1.5;
cond.ori_fraction = 0.4;
cond.temp_kernel = 'half-hamming';
cond.temp_bandwidth = 4;
cond.n_dirs = 16;
cond.ori_mix = 1;
cond.speed = 0.25;

params = stimulus.utils.factorize(cond);
fprintf('Total duration: %4.2f s\n', sum([params.duration]))

% generate conditions
hashes = control.makeConditions(stimulus.Monet2, params);

% push trials
control.pushTrials(hashes(randperm(numel(hashes))))
end
