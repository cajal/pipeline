%{
preprocess.SyncedMonet (computed) #  Monet stimuli synced to frame times
-> preprocess.Sync
-----
movie = null : longblob   #  movie synced to frame times
%}

classdef SyncedMonet < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = preprocess.Sync & preprocess.PrepareGalvo
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            monetTrials = preprocess.Sync*vis.Trial*vis.Monet*vis.MonetLookup ...
                & key & 'trial_idx between first_trial and last_trial';
            if count(monetTrials)
                caTimes = fetch1(preprocess.Sync & key, 'frame_times');
                nSlices = fetch1(preprocess.PrepareGalvo & key, 'nslices');
                caTimes = caTimes(1:nSlices:end);
                syncedMovie = [];
                for trialKey = fetch(monetTrials)'
                    % resample movies to the stimulus times.
                    [movie, flips] = fetch1(...
                        vis.Trial * vis.Monet * vis.MonetLookup & trialKey, ...
                        'cached_movie', 'flip_times');
                    sz = size(movie);
                    if isempty(syncedMovie)
                        syncedMovie = nan(sz(1), sz(2), length(caTimes));
                    end
                    timeInd = find(caTimes > flips(1) & caTimes < flips(end));
                    if ~isempty(timeInd)
                        syncedMovie(:,:,timeInd) = uint8(reshape(...
                            interp1(flips, reshape(double(movie), sz(1)*sz(2), sz(3))', caTimes(timeInd))', ...
                            sz(1), sz(2), length(timeInd)));                      %#ok<AGROW>
                    end                   
                end
                key.movie = syncedMovie;
            end
            self.insert(key)
        end
    end
    
end