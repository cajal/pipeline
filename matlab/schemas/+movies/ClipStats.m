%{
# Clip statistics
-> stimulus.MovieClip
---
frame_mean             : mediumblob      # frame mean
frame_std              : mediumblob      # frame standard deviation
frame_kurtosis         : mediumblob      # frame kurtosis
frame_diff             : mediumblob      # frame difference
frame_diff_low         : mediumblob      # frame difference from downsampled frames
center_mean             : mediumblob      # center frame mean
center_std              : mediumblob      # center frame standard deviation
center_kurtosis         : mediumblob      # center frame kurtosis
center_diff             : mediumblob      # center frame difference
center_diff_low         : mediumblob      # center frame difference from downsampled frames
%}

classdef ClipStats < dj.Imported
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            % get video file
            filename = export(stimulus.MovieClip & key);
            vidReader = VideoReader(filename{1});
            
            % initialize
            iframe = 0;
            nframes =  floor(vidReader.Duration*vidReader.FrameRate);
            Mov = nan(vidReader.Height,vidReader.Width,nframes);
            
            % fun for each frame
            while hasFrame(vidReader)
                iframe = iframe+1;
                frame =  readFrame(vidReader);
                Mov(:,:,iframe) = frame(:,:,1);
            end
            Mov = Mov(:,:,1:iframe);
            lmov = imresize(Mov,0.1);
            mov = reshape(permute(Mov,[3 1 2]),iframe,[]);
            lmov = reshape(permute(lmov,[3 1 2]),iframe,[]);
            dMov = diff(mov);
            dMov(end+1,:) = nan(1,size(mov,2));
            dlMov = diff(lmov);
            dlMov(end+1,:) = nan(1,size(lmov,2));
            
            key.frame_mean = mean(mov,2);
            key.frame_std = std(mov,[],2);
            key.frame_kurtosis = kurtosis(mov,[],2);
            key.frame_diff = mean(dMov,2);
            key.frame_diff_low = mean(dlMov,2);
            
            % center ~20deg in ~100deg coverage
            sz = round(size(Mov,2)/5/2);
            ct = round(size(Mov)/2);
            Mov = Mov(ct(1)-sz(1):ct(1)+sz,ct(2)-sz:ct(2)+sz,:);
            mov = reshape(permute(Mov,[3 1 2]),iframe,[]);
            dMov = diff(mov);
            dMov(end+1,:) = nan(1,size(mov,2));
            lmov = imresize(Mov,0.1);
            lmov = reshape(permute(lmov,[3 1 2]),iframe,[]);
            dlMov = diff(lmov);
            dlMov(end+1,:) = nan(1,size(lmov,2));
            key.center_mean = mean(mov,2);
            key.center_std = std(mov,[],2);
            key.center_kurtosis = kurtosis(mov,[],2);
            key.center_diff = mean(dMov,2);
            key.center_diff_low = mean(dlMov,2);
            
            insert( obj, key );
        end
    end
    
end