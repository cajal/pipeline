%{
# Optical Flow for all frames using Lucas-Kanade method
-> stimulus.MovieClip
---
frame_up                   : mediumblob   # positive X component of velocity
frame_right                : mediumblob   # positive Y component of velocity
frame_down                 : mediumblob   # negative X component of velocity
frame_left                 : mediumblob   # negative Y component of velocity
frame_orientation          : mediumblob   # mean phase angle of optical flow
frame_magnitude            : mediumblob   # Magnitude of optical flow
center_orientation         : mediumblob   # mean phase angle of optical flow
center_magnitude           : mediumblob   # Magnitude of optical flow
%}

classdef  OpticalFlow < dj.Imported
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            
            % get video file
            filename = export(stimulus.MovieClip & key);
            vidReader = VideoReader(filename{1});
            
            % construct optic flow algorithm
            opticFlow = opticalFlowLK('NoiseThreshold',0.004);

            % initialize
            iframe = 0;
            nframes =  floor(vidReader.Duration*vidReader.FrameRate);
            up = nan(nframes,1);
            Orientation = up; Magnitude = up; down = up; left = up; right = up; 
            cup = up;cOrientation = up; cMagnitude = up; cdown = up; cleft = up; cright = up; 
            
            % center
            sz = round(vidReader.Width/5/2);
            ct = round([vidReader.Height,vidReader.Width]/2);
            
            % fun for each frame
            while hasFrame(vidReader)
                iframe = iframe+1;
                frame = readFrame(vidReader);
                flow = estimateFlow(opticFlow,frame(:,:,1));
                Orientation(iframe) = nanmean(flow.Orientation(:));
                Magnitude(iframe) = nanmean(flow.Magnitude(:));
                up(iframe) =  nanmean(flow.Vx(flow.Vx<0));
                down(iframe) =  nanmean(flow.Vx(flow.Vx>0));
                left(iframe) =  nanmean(flow.Vy(flow.Vy<0));
                right(iframe) =  nanmean(flow.Vy(flow.Vy>0));
                
                Vx = flow.Vx(ct(1)-sz(1):ct(1)+sz,ct(2)-sz:ct(2)+sz);
                Vy = flow.Vy(ct(1)-sz(1):ct(1)+sz,ct(2)-sz:ct(2)+sz);
                Ori = flow.Orientation(ct(1)-sz(1):ct(1)+sz,ct(2)-sz:ct(2)+sz);
                Mag = flow.Magnitude(ct(1)-sz(1):ct(1)+sz,ct(2)-sz:ct(2)+sz);
                
                cOrientation(iframe) = nanmean(Ori(:));
                cMagnitude(iframe) = nanmean(Mag(:));
                cup(iframe) =  nanmean(Vx(Vx<0));
                cdown(iframe) =  nanmean(Vx(Vx>0));
                cleft(iframe) =  nanmean(Vy(Vy<0));
                cright(iframe) =  nanmean(Vy(Vy>0));
            end
            key.frame_up = single(up(1:iframe));
            key.frame_down = single(down(1:iframe));
            key.frame_left = single(left(1:iframe));
            key.frame_right = single(right(1:iframe));
            key.frame_orientation = single(Orientation(1:iframe));
            key.frame_magnitude =single( Magnitude(1:iframe));
            
%             key.center_up = single(cup(1:iframe));
%             key.center_down = single(cdown(1:iframe));
%             key.center_left = single(cleft(1:iframe));
%             key.center_right = single(cright(1:iframe));
            key.center_orientation = single(cOrientation(1:iframe));
            key.center_magnitude =single(cMagnitude(1:iframe));
            insert( obj, key );
        end
    end
    
end