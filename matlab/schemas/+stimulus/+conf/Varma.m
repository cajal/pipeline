function Varma
control = stimulus.getControl;
control.clearAll()   % clear trial queue and cached conditions.


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
paramsM = stimulus.utils.factorize(cond);
nblocks = 2;
fprintf('Total duration: %4.2f s\n', nblocks*(sum([paramsM.duration]) + sum([paramsM.pre_blank_period])))

% generate conditions
hashesM = control.makeConditions(stimulus.Matisse2, paramsM);

clear cond;

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
Dur_per_Cond    = 60; % duration of stimulus for each condition in s, including the repeated clips
Dur_RepClip     = 1.5; % duration of the tiny clips that are repeated in s
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
% cond.noise_seed         = 1:3;
cond.pattern_upscale    = 10;
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

% generate conditions
hashesVarma = control.makeConditions(stimulus.Varma, params);

% fprintf('Total duration of all conditions = %3.2f s\n', sum([params.pre_blank_period]) + sum([params.duration]))


fprintf('Total duration of all conditions = %3.2f s\n', sum([params.pre_blank_period]) + sum([params.duration]) + sum([paramsM.pre_blank_period]) + sum([paramsM.duration]))



% hashes = [hashesVarma(1:84); hashesM; hashesVarma(85:end)];
hashes = [hashesVarma(1:5); hashesM(1:10); hashesVarma(6:10)];

control.pushTrials(hashes);

% % push trials
% nblocks = 3;
% for i=1:nblocks
%     control.pushTrials(hashes(randperm(numel(hashes))))
% end
% end