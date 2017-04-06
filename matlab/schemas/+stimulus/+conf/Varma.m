function Varma
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions.

cond.fps = 60;
cond.pre_blank_period   = 1.0;
cond.noise_seed         = 1:30;
cond.pattern_upscale    = [5, 10];
cond.pattern_width      = 32;
cond.duration           = 3;
cond.pattern_aspect     = 1.7;
cond.gaborpatchsize     = 0.28; % changed from 0.34 to 0.28
cond.gabor_wlscale      = 4;
cond.gabor_envscale     = 6;
cond.gabor_ell          = 1;
cond.gaussfilt_scale    = 1;
cond.gaussfilt_istd     = 2;
cond.gaussfiltext_scale = 1;
cond.gaussfiltext_istd  = 2.4;
cond.filt_noise_bw       = 0.5;
cond.filt_ori_bw         = 0.5;
cond.filt_cont_bw        = 0.5;
cond.filt_gammshape     = 0.35;
cond.filt_gammscale     = 2;

% assert(isscalar(cond))
params = stimulus.utils.factorize(cond);

% generate conditions
hashes = control.makeConditions(stimulus.Varma, params);

% push trials
nblocks = 3;
for i=1:nblocks
    control.pushTrials(hashes(randperm(numel(hashes))))
end
end