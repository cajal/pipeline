%{
# Population analyis for each clip
-> fuse.ScanDone
-> stimulus.Clip
---
frame_mean             : mediumblob      # frame mean
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
            [Spikes, frame_times] = getAdjustedSpikes(fuse.ActivityTrace & key,'soma');
            trace_keys = fetch(fuse.ActivityTrace & key);
            [Traces, caTimes] = pipetools.getAdjustedSpikes(trace_keys);
            xm = min([length(caTimes) length(Traces)]);
            X = @(t) interp1(caTimes(1:xm)-caTimes(1), Traces(1:xm,:), t, 'linear', nan);  % traces indexed by time
            
            % fetch stuff
            flip_times = fetchn(stimulus.Trial & key,'flip_times');
            ft_sz = cellfun(@(x) size(x,2),flip_times);
            tidx = ft_sz>=prctile(ft_sz,99);
            flip_times = cell2mat(flip_times(tidx));
            
            % subsample traces
            fps = 1/median(diff(flip_times(1,:)));
            d = max(1,round(bin/1000*fps));
            traces = convn(permute(X(flip_times - caTimes(1)),[2 3 1]),ones(d,1)/d,'same');
            traces = traces(1:d:end,:,:);
            traces = permute(X(flip_times - caTimes(1)),[2 3 1]);
%             traces = trresize(traces,fps,bin,'linear');
            
            % split for unique stimuli
            for istim = 1:length(Stims)
                [s_trials,s_clips,s_names] = fetchn(trials*vis.MovieClipCond &...
                    sprintf('movie_name = "%s"',Stims{istim}),'trial_idx','clip_number','movie_name');
                [tr_idx, b]= ismember(trial_idxs,s_trials);
                st_idx = b(b>0);
                dat = permute(traces(:,:,tr_idx),[2 3 1]);
                info.bins{istim} = reshape(repmat(1:size(dat,3),size(dat,2),1),1,[]);
                info.trials{istim} = reshape(repmat(s_trials(st_idx),1,size(dat,3)),1,[]);
                info.clips{istim} = reshape(repmat(s_clips(st_idx),1,size(dat,3)),1,[]);
                info.names{istim} = reshape(repmat(s_names(st_idx),1,size(dat,3)),1,[]);
                Data{istim} = reshape(dat,size(traces,2),[]);
            end
        end
    end
    
end