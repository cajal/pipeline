%{
map.OptImageBar (imported) #
-> experiment.Scan
axis                        : enum('vertical', 'horizontal')# the direction of bar movement
---
amp                         : longblob                      # amplitude of the fft phase spectrum
ang                         : longblob                      # angle of the fft phase spectrum
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
                    
                case 'scanimage'
                    % get Optical data
                    disp 'loading movie...'
                    if strcmp(setup,'2P4') % mesoscope
                        path = getLocalPath(fullfile(path, sprintf('%s*.tif', name)));
                        reader = ne7.scanreader.readscan(path,'int16',1);
                        Data = permute(squeeze(mean(reader(:,:,:,:,:),1)),[3 1 2]);
                        data_fs = reader.fps;
                        nslices = reader.nScanningDepths;
                    else
                        reader = preprocess.getGalvoReader(key);
                        Data = squeeze(mean(reader(:,:,1,:,:),4));
                        Data = permute(Data,[3 1 2]);
                        [nslices, data_fs] = fetch1(preprocess.PrepareGalvo & key,'nslices','fps');
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
            
            params = getParams(params,varargin);
            
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
                if ~isempty(vessels); v = normalize(abs(normalize(vessels).^params.vcontrast));end
                
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
                    combined_map = hsv2rgb(cat(3,h,cat(3,s,v)));
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
            params = getParams(params,varargin);
            
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
            
            params = getParams(params,varargin);
            
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
        
                function [area_map, grad_diff, vessels] = extractAreas(obj,varargin)
            %%
            params.init_gauss = 10;
            params.grad_gauss = 0.1; % initial map gauss filter in sd parameter
            params.diff_gauss =10; % gradient diff map gauss filter in sd parameter
            params.diff_open = 10; % gradient diff imopen param
            params.sign_thr = 1; % threshold as standard deviations of signmap
            params.step = .01; % mask expand step parameter
            params.min_area_size = 1000;  % min area size in pixels
            params.final_erode = 10;
            params.final_dilate = 10;
            params.exp = 2;
            
            params = getParams(params,varargin);
            
            % find horizontal & verical map keys
            Hkeys = fetch(map.OptImageBar & (experiment.Session & obj) & 'axis="horizontal"');
            Vkeys = fetch(map.OptImageBar & (experiment.Session & obj) & 'axis="vertical"');
            
            % get vessels
            vessels = normalize(fetch1(map.OptImageBar & Hkeys(end),'vessels'));
            
            % fetch horizontal & vertical maps
            [H, A1] = fetch1(map.OptImageBar & Hkeys(end),'ang','amp');
            [Hor(:,:,1),Hor(:,:,2),Hor(:,:,3)] = plot(map.OptImageBar & Hkeys(end),'exp',params.exp,'sigma',params.init_gauss);
            [V,A2] = fetch1(map.OptImageBar & Vkeys(end),'ang','amp');
            [Ver(:,:,1),Ver(:,:,2),Ver(:,:,3)] = plot(map.OptImageBar & Vkeys(end),'exp',params.exp,'sigma',params.init_gauss);
            A = (A1+A2)/2;
            
            % calculate gradients
            %             [~,dH] = imgradient(H);
            %             [~,dV] = imgradient(V);
            [~,dH] = imgradient(Hor(:,:,1));
            [~,dV] = imgradient(Ver(:,:,1));
            
            % filter gradients
            dH = imgaussfilt(dH,params.grad_gauss);
            dV = imgaussfilt(dV,params.grad_gauss);
            grad_diff = imopen(imgaussfilt(sind(dH  - dV),params.diff_gauss),strel('disk',params.diff_open));
            
            % plot maps
            clf;%figure
            subplot(2,2,1)
            imagesc(hsv2rgb(Hor)); axis image; axis off; title('Horizontal Retinotopy')
            subplot(2,2,2)
            imagesc(hsv2rgb(Ver)); axis image; axis off; title('Vertical Retinotopy')
            subplot(2,2,3)
            imagesc(grad_diff); axis image; axis off; title('Visual Field Sign Map'); colormap jet
            
            %% filter & expand masks
            filtered_grad_diff = zeros(size(grad_diff));
            imStd = std(grad_diff(:));
            filtered_grad_diff(grad_diff>imStd*params.sign_thr)= 1;
            filtered_grad_diff(grad_diff<-imStd*params.sign_thr)= -1;
            SE = strel('arbitrary',[0 1 0;1 1 1;0 1 0]); %%% parameter
            SEMAX = strel('arbitrary',[0 0 1 0 0;0 1 1 1 0;1 1 1 1 1;0 1 1 1 0;0 0 1 0 0]); %%% parameter
            [all_areas, n] = bwlabel(filtered_grad_diff);
            for iThr = 1:-params.step:0.05
                n = max(all_areas(:));
                thr = imStd*iThr;
                for iArea = 1:n
                    one_area = imfill(imdilate(all_areas==iArea,SE),'holes');
                    all_areas(one_area & all_areas==0 & abs(grad_diff)>thr) = iArea;
                end
                lmax = imdilate(imbinarize(all_areas),SEMAX);
                one_area = bwlabel(abs(grad_diff)>thr & lmax == 0);
                stats = regionprops(one_area,'area');
                un = unique(one_area(:));
                for i = un(un>0)'
                    if stats(i).Area<params.min_area_size
                        one_area(one_area==i) = 0;
                    end
                end
                one_area = bwlabel(one_area);
                if any(one_area(:))
                    all_areas(logical(one_area)) = one_area(logical(one_area))+n;
                end
            end
            
            % select areas that have at least half their area with amplitude more than the threshold
            area_map = zeros(size(all_areas));
            idx = 0;
            thr = mean(A(:));% max(A(:)) - 2 * std(A(:)); %%% parameter
            for iarea = 1:n
                if ~(sum(all_areas(:)==iarea)/sum(all_areas(:)==iarea & A(:)>thr)>2)
                    idx = idx+1;
                    mask = imdilate(imerode(all_areas==iarea,strel('disk',params.final_erode)),strel('disk',params.final_dilate));
                    area_map(mask) = idx;
                end
            end
            
            % plot unique areas
            im(:,:,1) = normalize(area_map);
            im(:,:,2) = area_map>0;
            im(:,:,3) = vessels;
            subplot(2,2,4)
            image(hsv2rgb(im)); axis image; axis off; title('Areas')
            
            %% add numbers to areas
            stats = struct([]);
            for iarea = 1:max(area_map(:))
                s = regionprops(area_map==iarea,'area','Centroid');
                if iarea==1
                    stats = s;
                else
                    stats(iarea).Area = s.Area;
                    stats(iarea).Centroid = s.Centroid;
                end
                stats(iarea).sign = mean(grad_diff(area_map==iarea))>0;
                text(stats(iarea).Centroid(1),stats(iarea).Centroid(2),num2str(iarea))
            end
        end
        
        function area_map = editMask(self, masks, background)
            if nargin<3
                images = fetchn(self, 'vessels');
                background = images{1};
            end
            area_map = ne7.ui.paintMasks(background,masks);
            
            % add numbers to areas
            stats = struct([]);idx = 0;
            areas = unique(area_map(:));
            for iarea = areas(2:end)'
                idx = idx+1;
                s = regionprops(area_map==iarea,'area','Centroid');
                if isempty(stats)
                    stats = s;
                else
                    stats(idx).Area = s.Area;
                    stats(idx).Centroid = s.Centroid;
                end
                stats(idx).sign = mean(area_map(area_map==iarea))>0;
                text(stats(idx).Centroid(1),stats(idx).Centroid(2),num2str(idx))
            end
        end
        
        function insertAreas(obj,area_map,varargin)
            c = get(gca,'Children');
            istxt = find(arrayfun(@(x) strcmp(x.Type,'text'),c));
            p = []; t =[];
            for itxt = istxt'
                p(itxt,:) = c(itxt).Position;
                t{itxt}= c(itxt).String;
            end
            
            areas = fetchn(map.Area,'area');
            for iarea = 1:max(area_map(:))
                im = area_map==iarea;
                idx = im(sub2ind(size(area_map),round(p(:,2)),round(p(:,1))));
                if all(idx==0);continue;end
                midx = strcmp(areas,t{idx});
                if all(midx==0);continue;end
                key = fetch(experiment.Scan & obj);
                key.area = areas{midx};
                key.mask = im;
                key.slice = 1;
                insert(map.AreaMask,key)
            end
        end
        
    end
    
end