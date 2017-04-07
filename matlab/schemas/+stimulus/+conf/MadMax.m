function MadMax
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions.

cond.movie_name = 'MadMax';
cond.clip_number = 1:111;
cond.cut_after = 10;

% assert(isscalar(cond))
params = stimulus.utils.factorize(cond);

% generate conditions
hashes = control.makeConditions(stimulus.Clip, params);

% push trials
nblocks = 3;
for i=1:nblocks
    control.pushTrials(hashes(randperm(numel(hashes))))
end
end