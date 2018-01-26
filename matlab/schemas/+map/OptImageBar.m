%{
map.OptImageBar (imported) #
-> experiment.Scan
axis                        : enum('vertical', 'horizontal')# the direction of bar movement
---
amp                         : longblob                      # amplitude of the fft phase spectrum
ang                         : longblob                      # angle of the fft phase spectrum
pxpitch                     : double                        # pixel pitch of the map (microns per pixel)
vessels=null                : mediumblob                    #
%}


classdef OptImageBar < dj.Imported
    
    properties (Constant)
        keySource = (experiment.Scan & 'aim = "intrinsic" AND software="imager" OR aim="widefield"') - experiment.ScanIgnored
    end
    
    methods(Access=protected)
        
        function makeTuples( obj, key )
            
            % get frame times
            if ~exists(stimulus.Sync & key)
                disp 'Syncing...'
                populate(stimulus.Sync,key)
            end
            frame_times =fetch1(stimulus.Sync & key,'frame_times');
            
            % get scan info
            [name, path, software, setup] = fetch1( experiment.Scan * experiment.Session & key ,...
                'filename','scan_path','software','rig');
            switch software
                case 'imager'
                    % get Optical data
                    disp 'loading movie...'
                    [Data, data_fs] = getOpticalData(key); % time in sec
                    
                    % get the vessel image
                    disp 'getting the vessels...'
                    k = [];
                    k.session = key.session;
                    k.animal_id = key.animal_id;
                    k.site_number = fetch1(experiment.Scan & key,'site_number');
                    vesObj = experiment.Scan - experiment.ScanIgnored & k & 'software = "imager" and aim = "vessels"';
                    if ~isempty(vesObj)
                        keys = fetch( vesObj);
                        vessels = squeeze(mean(getOpticalData(keys(end))));
                    end
                    pxpitch = 3800/size(Data,2);
                case 'scanimage'
                    % get Optical data
                    disp 'loading movie...'
                    if strcmp(setup,'2P4') % mesoscope
                        path = getLocalPath(fullfile(path, sprintf('%s*.tif', name)));
                        reader = ne7.scanreader.readscan(path,'int16',1);
                        pxpitch = reader.fieldHeightsInMicrons(1)/reader.fieldHeights(1);
                        Data = permute(squeeze(mean(reader(:,:,:,:,:),1)),[3 1 2]);
                        data_fs = reader.fps;
                        nslices = reader.nScanningDepths;
                    else
                        reader = preprocess.getGalvoReader(key);
                        
                        Data = squeeze(mean(reader(:,:,1,:,:),4));
                        Data = permute(Data,[3 1 2]);
                        [nslices, data_fs] = fetch1(preprocess.PrepareGalvo & key,'nslices','fps');
                        
                        % calculate pxpitch
                        sz = size(reader);
                        px_height = sz(1);
                        slowScanMultiplier = reader.header.hRoiManager_scanAngleMultiplierSlow;
                        zoom = reader.zoom;
                        fov = experiment.FOV * pro(experiment.Session*experiment.Scan & key, 'rig', 'lens', 'session_date') & 'session_date>=fov_ts';
                        mags = fov.fetchn('mag');
                        [~, i] = min(abs(log(mags/zoom)));
                        mag = mags(i); % closest measured magnification
                        um_height = fetch1(fov & struct('mag', mag), 'height');
                        um_height = um_height * mag/zoom * slowScanMultiplier;
                        pxpitch = um_height/px_height;
                    end
                    
                    % fix frame times
                    frame_times = frame_times(1:nslices:end);
                    frame_times = frame_times(1:size(Data,1));
                    
                    % get the vessel image
                    vessels = squeeze(mean(Data(:,:,:)));
            end
            
            % DF/F
            mData = mean(Data);
            Data = bsxfun(@rdivide,bsxfun(@minus,Data,mData),mData);
            
            % loop through axis
            [axis,cond_idices] = fetchn(stimulus.FancyBar * (stimulus.Condition & (stimulus.Trial & key)),'axis','condition_hash');
            uaxis = unique(axis);
            for iaxis = 1:length(uaxis)
                
                key.axis = axis{iaxis};
                icond = [];
                icond.condition_hash = cond_idices{strcmp(axis,axis{iaxis})};
                
                % Get stim data
                times  = fetchn(stimulus.Trial * stimulus.Condition & key & icond,'flip_times');
                
                % find trace segments
                dataCell = cell(1,length(times));
                for iTrial = 1:length(times)
                    dataCell{iTrial} = Data(frame_times>=times{iTrial}(1) & ...
                        frame_times<times{iTrial}(end),:,:);
                end
                
                % remove incomplete trials
                tracessize = cell2mat(cellfun(@size,dataCell,'UniformOutput',0)');
                indx = tracessize(:,1) >= cellfun(@(x) x(end)-x(1),times)*9/10*data_fs;
                dataCell = dataCell(indx);
                tracessize = tracessize(indx,1);
                
                % equalize trial length
                dataCell = cellfun(@(x) permute(zscore(x(1:min(tracessize(:,1)),:,:)),...
                    [2 3 1]),dataCell,'UniformOutput',0);
                tf = data_fs/size(dataCell{1},3);
                dataCell = permute(cat(3,dataCell{:}),[3 1 2]);
                imsize = size(dataCell);
                
                % subtract mean for the fft
                dataCell = (bsxfun(@minus,dataCell(:,:),mean(dataCell(:,:))));
                
                T = 1/data_fs; % frame length
                L = size(dataCell,1); % Length of signal
                t = (0:L-1)*T; % time series
                
                % do it
                disp 'computing...'
                R = exp(2*pi*1i*t*tf)*dataCell;
                imP = squeeze(reshape((angle(R)),imsize(2),imsize(3)));
                imA = squeeze(reshape((abs(R)),imsize(2),imsize(3)));
                
                % save the data
                disp 'inserting data...'
                key.amp = imA;
                key.ang = imP;
                key.pxpitch = pxpitch;
                if ~isempty(vessels); key.vessels = vessels; end
                
                insert(obj,key)
            end
            disp 'done!'
        end
    end
    
    methods
        
        function  [iH, iS, iV] = plot(obj,varargin)
            
            % plot(obj)
            %
            % Plots the intrinsic imaging data aquired with Intrinsic Imager
            %
            % MF 2012, MF 2016
            
            params.sigma = 2; %sigma of the gaussian filter
            params.saturation = 1; % saturation scaling
            params.exp = []; % exponent factor of rescaling, 1-2 works
            params.shift = 0; % angular shift for improving map presentation
            params.subplot = true;
            params.vcontrast = 1;
            params.figure = [];
            
            params = ne7.mat.getParams(params,varargin);
            
            % define normalize function
            normalize = @(x) (x-min(x(:)))./(max(x(:)) - min(x(:)));
            
            % fetch all the keys
            keys = fetch(obj);
            if isempty(keys); disp('Nothing found!'); return; end
            
            for ikey = 1:length(keys)
                
                % get data
                [imP, vessels, imA] = fetch1(obj & keys(ikey),'ang','vessels','amp');
                
                % process image range
                imP(imP<-3.14) = imP(imP<-3.14) +3.14*2;
                imP(imP>3.14) = imP(imP>3.14) -3.14*2;
                uv =linspace(-3.14,3.14,20) ;
                n = histc(imP(:),uv);
                [~,i] = min(n(1:end-1)) ;
                minmode = uv(i);
                imP = imP+minmode+3.14;
                imP(imP<-3.14) = imP(imP<-3.14) +3.14*2;
                imP(imP>3.14) = imP(imP>3.14) -3.14*2;
                if ~isempty(params.exp)
                    imP = imP-nanmedian(imP(:));
                    imP(imP<-3.14) = imP(imP<-3.14) +3.14*2;
                    imP(imP>3.14) = imP(imP>3.14) -3.14*2;
                    imP = imP+params.shift;
                    imP(imP<0) = normalize(exp((normalize((imP(imP<0)))+1).^params.exp))-1;
                    imP(imP>0) =  normalize(-exp((normalize((-imP(imP>0)))+1).^params.exp));
                end
                imA(imA>prctile(imA(:),99)) = prctile(imA(:),99);
                
                % create the hsv map
                h = imgaussfilt(normalize(imP),params.sigma);
                s = imgaussfilt(normalize(imA),params.sigma)*params.saturation;
                v = ones(size(imA));
                if ~isempty(vessels) && params.vcontrast >0; v = normalize(abs(normalize(vessels).^params.vcontrast));end
                
                if nargout>0
                    iH{ikey} = h;
                    iS{ikey} = s;
                    iV{ikey} = v;
                else
                    if ~isempty(params.figure)
                        figure(params.figure);
                    else
                        figure;
                    end
                    set(gcf,'NumberTitle','off','name',sprintf(...
                        'OptMap direction:%s animal:%d session:%d scan:%d',...
                        keys(ikey).axis,keys(ikey).animal_id,keys(ikey).session,keys(ikey).scan_idx))
                    
                    % plot
                    angle_map = hsv2rgb(cat(3,h,cat(3,ones(size(s)),ones(size(v)))));
                    combined_map = hsv2rgb(cat(3,h,s,v));
                    if params.subplot
                        imshowpair(angle_map,combined_map,'montage')
                    else
                        imshow(angle_map)
                    end
                end
            end
            
            if ikey == 1 && nargout>0
                iH = iH{1};
                iS = iS{1};
                iV = iV{1};
            end
        end
        
        function plotTight(obj,varargin)
            
            params.saturation = 0.5;
            params = ne7.mat.getParams(params,varargin);
            
            keys = fetch(obj);
            if isempty(keys); disp('Nothing found!'); return; end
            
            for ikey = 1:length(keys)
                
                [h,s,v] = plot(obj & keys(ikey),params);
                
                % construct large image
                im = ones(size(h,1)*2,size(h,2)*2,3);
                im(1:size(h,1),1:size(h,2),:) = cat(3,zeros(size(v)),zeros(size(v)),v);
                im(size(h,1)+1:end,1:size(h,2),:) = cat(3,zeros(size(v)),zeros(size(v)),v);
                im(1:size(h,1),size(h,2)+1:end,:) = cat(3,h,s,v);
                im(size(h,1)+1:end,size(h,2)+1:end,:) =  cat(3,h,ones(size(h)),ones(size(h)));
                
                % plot
                figure
                set(gcf,'NumberTitle','off','name',sprintf(...
                    'OptMap direction:%s animal:%d session:%d scan:%d',...
                    keys(ikey).axis,keys(ikey).animal_id,keys(ikey).session,keys(ikey).scan_idx))
                imshow(hsv2rgb(im))
                
                % contour
                hold on
                contour(h,'showtext','on','linewidth',1,'levellist',0:0.05:1)
            end
        end
        
        function locateRF(obj,varargin)
            
            params.grad_gauss = 1;
            params.scale = 5;
            params.exp = 1;
            
            params = ne7.mat.getParams(params,varargin);
            
            % find horizontal & verical map keys
            Hkeys = fetch(map.OptImageBar & (experiment.Session & obj) & 'axis="horizontal"');
            Vkeys = fetch(map.OptImageBar & (experiment.Session & obj) & 'axis="vertical"');
            
            % fetch horizontal & vertical maps
            [Hor(:,:,1),Hor(:,:,2),Hor(:,:,3)] = plot(map.OptImageBar & Hkeys(end),'exp',params.exp);
            [Ver(:,:,1),Ver(:,:,2),Ver(:,:,3)] = plot(map.OptImageBar & Vkeys(end),'exp',params.exp);
            
            % get vessels
            vessels = normalize(Hor(:,:,3));
            
            % filter gradients
            H = roundall(normalize(imgaussfilt(Hor(:,:,1),params.grad_gauss))*params.scale,0.1);
            V = roundall(normalize(imgaussfilt(Ver(:,:,1),params.grad_gauss))*params.scale,0.1);
            
            % plot maps
            figure
            subplot(2,3,6)
            image(hsv2rgb(Ver)); axis image; axis off; title('Vertical Retinotopy')
            subplot(2,3,3)
            image(hsv2rgb(Hor)); axis image; axis off; title('Horizontal Retinotopy')
            hold on
            subplot(2,3,[1 2 4 5])
            hold on
            image(repmat(vessels,1,1,3)); axis image
            set(gca,'ydir','reverse')
            
            set (gcf, 'WindowButtonMotionFcn', {@mouseMove,H,V});
            
            function mouseMove (object, eventdata,H,V)
                global handles
                try
                    delete(handles)
                    
                    C = get (gca, 'CurrentPoint');
                    title(gca, ['(X,Y) = (', num2str(C(1,1)), ', ',num2str(C(1,2)), ')']);
                    Hv = H(round(C(1,2)),round(C(1,1)));
                    Vv = V(round(C(1,2)),round(C(1,1)));
                    I = find(H'==Hv & V'==Vv);
                    [x,y] = ind2sub(size(H),I);
                    handles = plot(x,y,'.r');
                end
            end
            
        end   

    end
    
end