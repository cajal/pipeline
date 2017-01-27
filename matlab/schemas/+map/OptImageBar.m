%{
map.OptImageBar (imported) #
-> experiment.Scan
axis                        : enum('vertical', 'horizontal')# the direction of bar movement
---
amp                         : longblob                      # amplitude of the fft phase spectrum
ang                         : longblob                      # angle of the fft phase spectrum
vessels=null                : mediumblob                    #
%}


classdef OptImageBar < dj.Relvar & dj.AutoPopulate
    
    properties (Constant)
        popRel = (experiment.Scan & 'aim = "intrinsic" OR software="imager" AND aim="widefield"') - experiment.ScanIgnored
    end
    
    methods(Access=protected)
        
        function makeTuples( obj, key )
            
            % get scan info
            [name, path, software] = fetch1( experiment.Scan * experiment.Session & key ,...
                'filename','scan_path','software');
            switch software
                case 'imager'
                    % get Optical data
                    disp 'loading movie...'
                    if isempty(strfind(name,'.h5')); name = [name '.h5'];end
                    [Data, data_fs,photodiode_signal, photodiode_fs] = ...
                        getOpticalData(getLocalPath(fullfile(path,name))); % time in sec
                    Data(:,end,:) = Data(:,end-1,:); % fix one missing line
                    fps = 30;   % does not need to be exact
                    disp 'synchronizing...'
                    % synchronize to stimulus
                    tuple =  sync(key, photodiode_signal, photodiode_fs, fps);
                    
                    % calculate frame times
                    frame_times = tuple.signal_start_time + ...
                        tuple.signal_duration*(1:size(Data,1))/size(Data,1);
                    
                    % import Sync table
                    tuple.frame_times = frame_times;
                    makeTuples(map.Sync,tuple)
                    
                    % get the vessel image
                    disp 'getting the vessels...'
                    k = [];
                    k.session = key.session;
                    k.animal_id = key.animal_id;
                    k.site_number = fetch1(experiment.Scan & key,'site_number');
                    vesObj = experiment.Scan & k & 'software = "imager" and aim = "vessels"';
                    if ~isempty(vesObj)
                        name = fetch1( vesObj ,'filename');
                        if isempty(strfind(name,'.h5')); name = [name '.h5'];end
                        filename = getLocalPath(fullfile(path,name));
                        vessels = squeeze(mean(getOpticalData(filename)));
                    end
                    
                case 'scanimage'
                    
                    % get Optical data
                    disp 'loading movie...'
                    reader = preprocess.getGalvoReader(key);
                    Data = squeeze(mean(reader(:,:,1,:,:),4));
                    Data = permute(Data,[3 1 2]);
                    [nslices, data_fs] = fetch1(preprocess.PrepareGalvo & key,'nslices','fps');
                    
                    % calculate frame times
                    frame_times = fetch1(preprocess.Sync & key,'frame_times');
                    frame_times = frame_times(1:nslices:end);
                    
%                     % get the vessel image
%                     disp 'getting the vessels...'
%                     k = [];
%                     k.session = key.session;
%                     k.animal_id = key.animal_id;
%                     k.site_number = fetch1(experiment.Scan & key,'site_number');
%                     ves_key = fetch(experiment.Scan & k & 'aim = "vessels"');
%                     reader = preprocess.getGalvoReader(ves_key);
%                     vessels = reader(:,:,:,:,:);
%                     vessels = mean(vessels(:,:,:),3);
                    vessels = [];
            end
            
            % DF/F
            mData = mean(Data);
            Data = bsxfun(@rdivide,bsxfun(@minus,Data,mData),mData);
            
            % loop through axis
            [axis,cond_idices] = fetchn(vis.FancyBar * vis.ScanConditions & key,'axis','cond_idx');
            uaxis = unique(axis);
            for iaxis = 1:length(uaxis)
                
                key.axis = axis{iaxis};
                icond = [];
                icond.cond_idx = cond_idices(strcmp(axis,axis{iaxis}));
                
                % Get stim data
                times  = fetchn(vis.Trial * vis.ScanConditions & key & icond,'flip_times');
                
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
            params.exp = []; % exponent factor of rescaling
            params.reverse = 0; % reverse the axis
            params.subplot = [1 2];
            params.amp = 0;
            params.shift = 0;
            params.vessels = false;
            params.figure = [];
            
            params = getParams(params,varargin);
            
            keys = fetch(obj);
            
            for ikey = 1:length(keys)
                subs = 1;
                if params.vessels;subs = 2;end
                
                [imP, vessels, imA] = fetch1(obj & keys(ikey),'ang','vessels','amp');
                
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
                    imP = imP-median(imP(:));
                    imP(imP<-3.14) = imP(imP<-3.14) +3.14*2;
                    imP(imP>3.14) = imP(imP>3.14) -3.14*2;
                    imP = imP+params.shift;
                    imP(imP<0) = normalize(exp((normalize((imP(imP<0)))+1).^params.exp))-1;
                    imP(imP>0) =  normalize(-exp((normalize((-imP(imP>0)))+1).^params.exp));
                end
                
                imA(imA>prctile(imA(:),99)) = prctile(imA(:),99);
                h = normalize(imP);
                s = ones(size(imP));
                if params.amp
                    v = normalize(imA);
                else
                    v = ones(size(imA));
                end
                s2 = normalize(imA);
                if ~isempty(vessels)
                    v2 = normalize(vessels);
                else
                    v2 = ones(size(imA));
                end
                
                if nargout>0
                    iH{ikey} = h;
                    iS{ikey} = s2;
                    iV{ikey} = v2;
                else
                    if isempty(params.figure)
                        figure
                    else
                        figure(params.figure)
                    end
                    set(gcf,'position',[50 200 920 435])
                    set(gcf,'name',['OptMap: ' num2str(keys(ikey).animal_id) '_' num2str(keys(ikey).session) '_' num2str(keys(ikey).scan_idx)])
                    
                    if any(params.subplot==1) && any(params.subplot==2)
                        subplot(subs,2,1)
                    end
                    if any(params.subplot==1)
                        im = (hsv2rgb(cat(3,h,cat(3,s,v))));
                        im = imgaussian(im,params.sigma);
                        imshow(im)
                        if params.reverse
                            set(gca,'xdir','reverse')
                        end
                    end
                    
                    if any(params.subplot==1) && any(params.subplot==2)
                        subplot(subs,2,2)
                    end
                    
                    if any(params.subplot==2)
                        s2 = imgaussian(s2,params.sigma);
                        h = imgaussian(h,params.sigma);
                        im = (hsv2rgb(cat(3,h,cat(3,s2,v2))));
                        
                        imshow(im)
                    end
                    
                    if params.vessels
                        subplot(subs,2,3:4)
                        imagesc(v2);
                        axis image
                        axis off
                        colormap gray
                    end
                    if params.reverse
                        set(gca,'xdir','reverse')
                    end
                    if ikey ~= length(keys)
                        pause
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
            
            params = [];
            params = getParams(params,varargin);
            
            [h,s,v] = plot(obj,params);
            
            im = ones(size(h,1)*2,size(h,2)*2,3);
            
            im(1:size(h,1),1:size(h,2),1) = h;
            im(1:size(h,1),1:size(h,2),2) = s;
            im(1:size(h,1),1:size(h,2),3) = v;
            
            im(size(h,1)+1:end,1:size(h,2),1) = zeros(size(v));
            im(size(h,1)+1:end,1:size(h,2),2) = zeros(size(v));
            im(size(h,1)+1:end,1:size(h,2),3) = v;
            
            im(1:size(h,1),size(h,2)+1:end,1) = h;
            im(1:size(h,1),size(h,2)+1:end,2) = ones(size(h));
            im(1:size(h,1),size(h,2)+1:end,3) = ones(size(h));
            
            im(size(h,1)+1:end,size(h,2)+1:end,1) = zeros(size(v));
            im(size(h,1)+1:end,size(h,2)+1:end,2) = zeros(size(v));
            im(size(h,1)+1:end,size(h,2)+1:end,3) = ones(size(h));
            
            figure
            imshow(hsv2rgb(im))
        end
    end
    
end