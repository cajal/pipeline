%{
# Average filterStatistics
-> stimulus.MovieClip
---
kurtosis               : float      # mean filter kurtosis
mean                   : float      # mean filter responses
std                    : float      # mean std of filters
sparseness             : float      # mean popsparseness of filters
nmean                  : float      # mean normalized filter responses
pcorr                  : float      # mean filter correlation
%}

classdef SimRespAvg < dj.Imported
    
    properties
        keySource = stimulus.MovieClip & movies.SimResp
    end
    
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            
            % fetch
            Resp = fetchn(movies.SimResp & key,'resp');
            Resp = cell2mat(Resp');
            idx = 1:10*60;
            Resp = abs(Resp(idx,:).^0.4);
            
            % compute pixel
            key.kurtosis = nanmean(kurtosis(Resp));
            key.std = nanmean(std(Resp));
            key.mean = nanmean(Resp(:));
            key.sparseness = nanmean(sparseness(Resp'));
            key.nmean = mean(mean(bsxfun(@rdivide,Resp,0.01+mean(Resp,2))));
            key.pcorr = nanmean(nanmean(corr(Resp)));
            
            % insert
            insert( obj, key );
        end
    end
    
end