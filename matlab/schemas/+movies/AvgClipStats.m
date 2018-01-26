%{
# Average clip statistics
-> stimulus.MovieClip
---
mean_kurtosis               : float      # mean frame kurtosis
mean_std                    : float      # mean frame std
std_mean                    : float      # std of frame means
mean_diff                   : float      # mean frame diff
mean_diff_low               : float      # mean frame diff with low pass
center_mean_kurtosis        : float      # mean center frame kurtosis
center_mean_std             : float      # mean center frame std
center_std_mean             : float      # standard deviation of frame center means
center_mean_diff            : float      # mean center frame diff
center_mean_diff_low        : float      # mean center frame diff with low pass
mean_up_of                  : float        # mean frame up velocity
mean_right_of               : float        # mean frame right velocity
mean_down_of                : float        # mean frame down velocity
mean_left_of                : float        # mean frame left velocity
std_ori_of                  : float        # std of phase angle of optical flow
mean_mag_of                 : float        # mean phase angle of optical flow
center_mean_mag_of          : float        # average frame center optic flow
center_std_ori_of           : float        # std of center phase angle of optical flow
%}

classdef AvgClipStats < dj.Imported
    
    properties
        keySource = stimulus.MovieClip & movies.ClipStats & movies.OpticalFlow
    end
    
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            
            % fetch
            [fm, fs, fk, fd, fdl, cm, cs, ck, cd, cdl] = fetch1(movies.ClipStats & key,...
                'frame_mean',...
                'frame_std',...
                'frame_kurtosis',...
                'frame_diff',...
                'frame_diff_low',...
                'center_mean',...
                'center_std',...
                'center_kurtosis',...
                'center_diff',...
                'center_diff_low');
            
            idx = 1:10*60;
            
            % compute pixel
            key.mean_kurtosis = nanmean(fk(idx));
            key.mean_std = nanmean(fs(idx));
            key.std_mean = nanstd(fm(idx));
            key.mean_diff = nanmean(fd(idx));
            key.mean_diff_low = nanmean(fdl(idx));
            key.center_mean_kurtosis = nanmean(ck(idx));
            key.center_mean_std = nanmean(cs(idx));
            key.center_std_mean = nanstd(cm(idx));
            key.center_mean_diff = nanmean(cd(idx));
            key.center_mean_diff_low = nanmean(cdl(idx));
            
            [fu,fr,fd,fl,fo,fm,co,cm] = fetch1(movies.OpticalFlow & key,...
                'frame_up',...
                'frame_right',...
                'frame_down',...
                'frame_left',...
                'frame_orientation',...
                'frame_magnitude',...
                'center_orientation',...
                'center_magnitude');
            
            % compute optical flow
            key.mean_up_of = nanmean(fu(idx));
            key.mean_right_of = nanmean(fr(idx));
            key.mean_down_of = nanmean(fd(idx));
            key.mean_left_of = nanmean(fl(idx));
            key.std_ori_of = nanstd(fo(idx));
            key.mean_mag_of = nanmean(fm(idx));
            key.center_mean_mag_of = nanmean(cm(idx));
            key.center_std_ori_of = nanstd(co(idx));
            
            % insert
            insert( obj, key );
        end
    end
    
end