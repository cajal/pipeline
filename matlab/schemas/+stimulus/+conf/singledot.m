function singledot
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions

params = struct(...
    'bg_level',    255,         ... (0-255) the index of the background luminance
    'dot_level',   0,           ... (0-255) the index of the dot luminance
    'dot_xsize',   0.125,        ... (fraction of the x length) size of dot x
    'dot_ysize',   0.125,        ... (fraction of the x length) size of dot y
    'dot_x',       linspace(-0.5,0.5,8),       ... (fraction of the x length, -0.5 to 0.5) position of dot on x axis
    'dot_y',       linspace(-0.28,0.28,5),      ... (fraction of the x length) position of dot on y axis
    'dot_shape',   'oval',       ... shape of dot, rect or oval
    'dot_time',    0.25           ... (secs) time each dot persists
    );

% generate conditions
assert(isscalar(params))
params = stimulus.utils.factorize(params);

% save conditions
hashes = control.makeConditions(stimulus.SingleDot, params);

% queue trials 
nblocks = 10;
fprintf('Total time: %g s\n', nblocks*(sum([params.dot_time])))
for i=1:nblocks
    control.pushTrials(hashes(randperm(numel(hashes))))
end
end
