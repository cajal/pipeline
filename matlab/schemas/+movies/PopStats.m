%{
# Population analyis for each clip
-> fuse.ScanDone
-> stimulus.Clip
---
act_mean             : float      #  mean activity
%}

classdef PopStats < dj.Imported
    
    methods
        keySource = aggr(fuse.ScanDone * stimulus.Clip & (stimulus.Movie & 'movie_class="cinema"'), stimulus.Trial, 'count(*)->n') & 'n>=2'
    end
    
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            
            
            
        end
    end
    
    methods
        function Data = getData(self,key,bin)
            
            if nargin<3
                bin = fetch1(obj.PopStatOpt & key, 'binsize');
            end

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
            d = max(1,round(bin/1000*fps));
            Data = convn(permute(X(flip_times - caTimes(1)),[2 3 1]),ones(d,1)/d,'same');
            Data = Data(1:d:end,:,:);            
        end
    end
    
end