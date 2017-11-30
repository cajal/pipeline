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
            Data = permute(Data(1:d:end,:,:),[2 1 3]);
        end
    end
end