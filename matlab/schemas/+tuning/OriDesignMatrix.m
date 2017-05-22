%{
# design matrix for directional response
-> tuning.Directional
-> tuning.CaKernel
---
design_matrix               : longblob                      # times x nConds
regressor_cov               : longblob                      # regressor covariance matrix,  nConds x nConds
%}


classdef OriDesignMatrix < dj.Computed

	properties(Constant)
		keySource = tuning.Directional*tuning.CaKernel & 'kernel=0'
	end

	methods(Access=protected)

		function makeTuples(self, key)
            caTimes = fetch1(preprocess.Sync & key, 'frame_times');
            trials = fetch(tuning.DirectionalTrial & key, 'onset', 'offset', 'direction');
            opt = fetch(tuning.CaKernel & key, '*');
            disp 'constructing design matrix...'
            G = tuning.OriDesignMatrix.makeDesignMatrix(caTimes, trials, opt);            
            key.design_matrix = single(G);
            key.regressor_cov = single(G'*G);
            self.insert(key)
        end
        
    end
    
    
    methods(Static)
        function G = makeDesignMatrix(times, trials, opt)
            % compute the directional tuning design matrix with a separate
            % regressor for each direction.
            
            alpha = @(x,a) (x>0).*x/a/a.*exp(-x/a);  % response shape
            
            % relevant trials
            [directions,~,condIdx] = unique([trials.direction]);
            assert(directions(1) == 0 && length(directions) >= 8 && ...
                all(abs(diff(diff(directions)))<1e-6), ...
                'motion directions must be uninformly distributed around the circle')
            
            G = zeros(length(times), length(unique(condIdx)), 'single');
            for iTrial = 1:length(trials)
                trial = trials(iTrial);
                switch opt.transient_shape
                    case 'onAlpha'
                        ix = find(times >= trial.onset & times < trial.onset+6*opt.tau);
                        G(ix, condIdx(iTrial)) = G(ix, condIdx(iTrial)) ...
                            + alpha(times(ix)-trial.onset,opt.tau)';
                    case 'exp'
                        ix = find(times>=trial.onset & times < trial.offset);
                        G(ix, condIdx(iTrial)) = G(ix, condIdx(iTrial)) ...
                            + 1 - exp((trial.onset-times(ix))/opt.tau)';
                        ix = find(times>=trial.offset & times < trial.offset+5*opt.tau);
                        G(ix, condIdx(iTrial)) = G(ix, condIdx(iTrial)) ...
                            + (1-exp((trial.onset-trial.offset)/opt.tau))*exp((trial.offset-times(ix))/opt.tau)';
                    otherwise
                        assert(false)
                end
            end
        end
    end
end
