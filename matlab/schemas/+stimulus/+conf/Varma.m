function Varma
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions.

cond.fps = 60;
cond.duration = 30;
cond.rng_seed = 1:60; % to be changed to 1:60 to generate Monet stim for 1800 s
cond.pattern_width = 72;
cond.pattern_aspect = 1.7;
cond.ori_coherence = 1.5;
cond.ori_fraction = 0.4;
cond.temp_kernel = 'half-hamming';
cond.temp_bandwidth = 4;
cond.n_dirs = 16;
cond.ori_mix = 1;
cond.speed = 0.25;

paramsM = stimulus.utils.factorize(cond);

clear cond

% Stimulus sequence: 
% S and L correpond to one major segment for short and long range
% correlation conditions
% The sequence we want is S, L, S, L, S, L, S, L (4 reps of major segments)
% Now, within each major segment, we have 5 repetitions each of the same 
% 's' and 'l' conditions: So S would comprise: s x 5, l x 5 interleaved in
% a random order
% Finally have to insert a Monet stimulus also in the middle

rng(0); % initial use just to generate the sequence

% Parameters
NStimConds      = 2; % No. of stimulus conditions. Together they comprise one major segment
TotalReps       = 4; % No. of repetitions of major segments
Dur_per_Cond    = 600; % duration of stimulus for each condition in s, including the repeated clips
Dur_RepClip     = 15; % duration of the tiny clips that are repeated in s
No_Reps         = 5;  % no. of repetitions
Dur_exc_reps    = Dur_per_Cond - NStimConds*Dur_RepClip*No_Reps; % duration excluding the repetitions

FullSeq = [];

for nn = 1:NStimConds*TotalReps
    X       = rand(NStimConds*No_Reps + 1,1);
    X       = round(10*Dur_exc_reps*X/sum(X))/10;
    X(end)  = X(end) - (sum(X) - Dur_exc_reps);
    RepPart = [Dur_RepClip*ones(NStimConds*No_Reps,1), X(2:end)];
    Seq  = reshape(RepPart',2*No_Reps*NStimConds,1);
    Seq  = [X(1); Seq];
    FullSeq = [FullSeq; Seq];
end

% make the vector of the required parameters
% make the vector of noise seeds and parameters


FixedSeeds      = 500 + 100*(0:NStimConds-1)';
ParamValues     = [0.5,1.5]; %0.5 for short range, 1.5 for long range

NoiseSeedVec    = (1:length(FullSeq))';
L               = 2*NStimConds*No_Reps + 1;
NoiseSeedMat    = reshape(NoiseSeedVec,L,NStimConds*TotalReps);
ParamMat        = repmat(ParamValues,1,TotalReps);
ParamMat        = repmat(ParamMat,L,1);

for nn = 1:NStimConds*TotalReps
    SeedVecTemp             = NoiseSeedMat(:,nn);
    TempVec                 = repmat(FixedSeeds,No_Reps,1);
    idx                     = randperm(length(TempVec));
    TempVec                 = TempVec(idx);
    SeedVecTemp(2:2:end)    = TempVec;
    NoiseSeedMat(:,nn)      = SeedVecTemp;
    
    ParamVecTemp            = ParamMat(:,nn);
    TempVec                 = repmat(ParamValues',No_Reps,1);
    TempVec                 = TempVec(idx);
    ParamVecTemp(2:2:end)   = TempVec;
    ParamMat(:,nn)          = ParamVecTemp;
end

NoiseSeedVec = NoiseSeedMat(:);
ParamVec     = ParamMat(:);


cond.fps = 60;
cond.pre_blank_period   = 0;
cond.noise_seed         = 1:length(NoiseSeedVec);
cond.pattern_upscale    = 3;
cond.pattern_width      = 32;
cond.duration           = 0;
cond.pattern_aspect     = 1.7;
cond.gaborpatchsize     = 0.28; 
cond.gabor_wlscale      = 4;
cond.gabor_envscale     = 6;
cond.gabor_ell          = 1;
cond.gaussfilt_scale    = 0;
cond.gaussfilt_istd     = 0.5;
cond.gaussfiltext_scale = 1;
cond.gaussfiltext_istd  = 1;
cond.filt_noise_bw      = 0.5;
cond.filt_ori_bw        = 0.5;
cond.filt_cont_bw       = 0.5;
cond.filt_gammshape     = 0.35;
cond.filt_gammscale     = 2;

% assert(isscalar(cond))
params = stimulus.utils.factorize(cond);

for kk = 1:length(NoiseSeedVec)
    params(kk).noise_seed = NoiseSeedVec(kk);
    params(kk).duration   = FullSeq(kk);
    params(kk).gaussfilt_scale = ParamVec(kk); 
end


fprintf('Total duration of all conditions = %3.2f s\n', sum(sum([params.duration]) + sum([paramsM.duration])));

% generate conditions
hashesMonet = control.makeConditions(stimulus.Monet2, paramsM);
hashesVarma = control.makeConditions(stimulus.Varma, params);

hashes = [hashesVarma(1:84); hashesMonet; hashesVarma(85:end)];

control.pushTrials(hashes);
