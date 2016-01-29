%{
rf.OriDesign (computed) # design matrix for directional tuning
-> rf.Sync
-> rf.SpaceTime
-----
ndirections     : tinyint    # number of directions
design_matrix   : longblob   # times x nConds
regressor_cov   : longblob   # regressor covariance matrix,  nConds x nConds
%}

cclassdef OriDesign < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = rf.Sync*rf.SpaceTime & rf.GratingResponses %  rf.GratingResponses should already be computed, which populates rf.SpaceTime
    end
    
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            tuple = key;
            times = fetch1(rf.Sync & key, 'frame_times');
            if key.spatial_freq==-1
                key = rmfield(key,'spatial_freq');
            end
            if key.temp_freq==-1
                key = rmfield(key,'temp_freq');
            end
            trialRel = rf.Sync*psy.Trial*psy.Grating & key & ...
                'trial_idx between first_trial and last_trial';
            disp 'constructing design matrix...'
            G = makeDesignMatrix(times, trialRel);
            
            tuple.ndirections = size(G,2);
            tuple.design_matrix = single(G);
            tuple.regressor_cov = single(G'*G);
            self.insert(tuple)
        end
    end
end


function G = makeDesignMatrix(times, trials)
% compute the directional tuning design matrix with a separate
% regressor for each direction.

tau = 0.8;
alpha = @(x) (x>0).*x/tau/tau.*exp(-x/tau);  % response shape

% relevant trials
if ~isstruct(trials)
    trials = fetch(trials, 'direction', 'flip_times');
end
[~,~,condIdx] = unique([trials.direction]);

G = zeros(length(times), length(unique(condIdx)), 'single');
for iTrial = 1:length(trials)
    trial = trials(iTrial);
    onset = trial.flip_times(2);  % second flip is the start of the drifting phase
    ix = find(times >= onset & times < onset+6*tau);
    G(ix, condIdx(iTrial)) = G(ix, condIdx(iTrial)) ...
        + alpha(times(ix)-onset)';    
end
end