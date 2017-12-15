%{
# Population analyis for each clip
-> fuse.ScanDone
-> stimulus.Clip
-> movies.PopStatsOpt
---
mean             : float      #  mean activity
reliability      : float      #  variance explained
mi               : float      #  mutual information - pairwize
rmi              : float      #  randomized mutual information
ntrials          : tinyint    #  number of trials
%}

classdef PopStats < dj.Imported
    
    properties
        keySource = aggr( movies.PopStatsOpt * fuse.ScanDone * stimulus.Clip & ...
            (stimulus.Movie & 'movie_class="cinema"'), stimulus.Trial, 'count(*)->n') & 'n>=2'
    end
    
    methods(Access=protected)
        function makeTuples(self,key) %create clips
            
            % get data
            Data = getData(movies.PopStats,key);
            
            tuple = key;
            tuple.ntrials = size(Data,3);
            
            % randomize Data
            rData = Data(:,:);
            ridx = randperm(size(rData,2));
            for i  = 1:1000;ridx = ridx(randperm(size(rData,2)));end
            rData = reshape(rData(:,ridx),size(Data));
            
            % mutual information
            tuple.mi = nnclassRawSV(Data);
            tuple.rmi = nnclassRawSV(rData);
            
            % reliability
            tuple.reliability  = reliability(Data);
            
            % mean
            tuple.mean = mean(Data(:));
            
            % insert
            self.insert(tuple)
        end
    end
    
    methods
        function Data = getData(~,key)
            
            % get traces
            [Traces, caTimes] = getAdjustedSpikes(fuse.ActivityTrace & key,'soma');
            X = @(t) interp1(caTimes-caTimes(1), Traces, t, 'linear', nan);  % traces indexed by time
            
            % fetch stuff
            flip_times = fetchn(stimulus.Trial & key,'flip_times');
            ft_sz = cellfun(@(x) size(x,2),flip_times);
            tidx = ft_sz>=prctile(ft_sz,99);
            flip_times = cell2mat(flip_times(tidx));
             
            % subsample traces
            fps = 1/median(diff(flip_times(1,:)));
            d = max(1,round(key.bin/1000*fps));
            Data = convn(permute(X(flip_times - caTimes(1)),[2 3 1]),ones(d,1)/d,'same');
            Data = permute(Data(1:d:end,:,:),[2 1 3]); % in [cells bins repeats]
        end
        
        function mi = nnclassRawSV(traces,varargin)
            % function [CA,CR,FP,FN] = nnclassRawSV(traces)
            %
            % performs a support vector machine classification
            % and outputs mutual information
            % plus the false positives,false negatives,correct acceptance and
            % correct rejections.
            % traces: [cells classes trials]

            % get the sizes
            ntrials = size(traces,3);
            
            % initialize
            pairs = nchoosek(1:size(traces,2),2);
            mi = cell(size(pairs,1),1);
            nclasses = 2;
            
            % loop through the pairs
            parfor ipair = 1:size(pairs,1)
                data = traces(:,pairs(ipair,:),:);
                
                % initialize
                F = zeros(nclasses);
                [CA,CR,FP,FN] = initialize('zeros',nclasses,1);
                
                % loop through trials
                for iTrial = 1:ntrials
                    
                    % calculate mean without taking that trial into account
                    ind = true(ntrials,1);
                    ind(iTrial) = false;
                    r = data(:,:,ind);
                    group = repmat((1:nclasses)',1,size(r,3));
                    SVMStruct = fitclinear(r(:,:)',group(:));
                    
                    % loop through classes
                    for iClass = 1:nclasses
                        %              indx = svmclassify(SVMStruct,data(:,iClass,iTrial)');
                        indx = predict(SVMStruct,data(:,iClass,iTrial)');
                        F(iClass,indx) = F(iClass,indx) + 1;
                    end
                end
                
                % loop through classes
                d = diag(F,0);
                for iclass = 1:nclasses
                    CA(iclass) = F(iclass,iclass);
                    dind = true(size(d));dind(iclass) = false;
                    CR(iclass) = sum(d(dind));
                    FN(iclass) = sum(F(iclass,dind));
                    FP(iclass) = sum(F(dind,iclass));
                end
                CM = zeros(2,2);
                CM(1,1) = sum(CA);
                CM(1,2) = sum(FN);
                CM(2,1) = sum(FP);
                CM(2,2) = sum(CR);
                
                p = CM/sum(CM(:));
                pi = sum(CM,2)/sum(CM(:));
                pj = sum(CM,1)/sum(CM(:));
                pij = pi*pj;
                if FN+FP == 0
                    mi{ipair} = 1;
                elseif CA+CR == 0
                    mi{ipair} = 0;
                else
                    mi{ipair} = sum(sum(p.*log2(p./pij)));
                end
            end
            
            mi = mean(cell2mat(mi));
            
        end
        
        function rl = reliability(traces)
            % function rl = reliability(traces)
            %
            % Computes the variance explained:
            % Reliability = True variance / Observed Variance
            % Observed variance = True Variance + Error Variance
            %
            % x_ij = ?_i + ?_ij
            %
            % Var_ij[x] = Var_i[?] + Var_ij[?]
            %
            % VE = Var[?]/Var[x]
            %
            % traces can be:
            % [cells time trials] or
            % {cells,time}(trials)
            
            % intitialize
            rl = nan(size(traces,1),1);
            if iscell(traces); sz = cellfun(@length,traces); end
            
            % loop through cells
            for icell = 1:size(traces,1)
                
                
                if iscell(traces)
                    trace = traces(icell,:);
                    for istim = 1:length(trace)
                        trace{istim}(end+1:max(sz(1,:))) = nan;
                        trace{istim} = trace{istim}(:);
                    end
                    trace = cell2mat(trace);
                else
                    if size(traces,3)<2;rl(icell)=nan;continue;end
                    trace = squeeze(traces(icell,:,:))'; %[trials time]
                end
                
                % filter trials
                trace = trace(:,sum(~isnan(trace))>1); % has at least 2 trials
                
                % explained Variance
                rl(icell) = var(nanmean(trace,1))/nanvar(trace(:));
            end
            
            % average the VE across the cells of one site
            rl = nanmean(rl);
        end
    end
end