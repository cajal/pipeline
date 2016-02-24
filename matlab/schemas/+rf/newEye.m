function newEye(key)

    p = fetch1(rf.Session & key,'hd5_path');
    f = fetch1(rf.Session & key,'file_base');
    n = fetch1(rf.Scan & key,'file_num');
    f = ['C:' p '/' f  num2str(n) 'eyetracking.avi']
    f = regexprep(f,{'/scratch01/','\/'},'\\');
    
    dat = rf.readHD5(key);
    packetLen = 2000;
    datT = pipetools.ts2sec(dat.ts, packetLen);
    eyeT = pipetools.ts2sec(dat.cam2ts, packetLen);
    
    totalFrames = length(eyeT);
    
    nFrames = 10;
    frameInd = round(linspace(1,totalFrames,nFrames));
    
    eyeObj = VideoReader(f);
    if totalFrames ~= get(eyeObj,'NumberOfFrames')
        disp([num2str(totalFrames) ' timestamps, but ' num2str(get(eyeObj,'NumberOfFrames')) ' movie frames.'])
        if totalFrames > get(eyeObj,'NumberOfFrames') && totalFrames && get(eyeObj,'NumberOfFrames')
            totalFrames = get(eyeObj,'NumberOfFrames');
            eyeT = eyeT(1:totalFrames);
            nFrames = 10;
            frameInd = round(linspace(1,totalFrames,nFrames));
        else
            error('Can not reconcile frame count')
        end
    end
    
    for j=1:nFrames
        frame = read(eyeObj,frameInd(j));
        frames(:,:,j) = squeeze(frame(:,:,1,1));
    end
    
    
    figure(3)
    imagesc(mean(frames,3))
    colormap('gray')
    axis image
    
    hRect=imrect(gca);
    
    pause
    
    roi = round(getPosition(hRect));
    roi = [roi(1) roi(1)+roi(3) roi(2) roi(2)+roi(4)];
    
    key.eye_time = eyeT;
    key.eye_roi = roi;
    insert(rf.Eye, key);
    
    clf
end