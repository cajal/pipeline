%{
rf.NoiseMap (computed) # my newest table
-> rf.Sync
-> rf.Trace
-> rf.NoiseMapMethod
-----
rf_nbins                    : smallint                      # temporal bins
rf_bin_size                 : float                         # (ms) temporal bin size
degrees_x                   : float                         # degrees along x
degrees_y                   : float                         # degrees along y
rf_map                      : longblob                      # receptive field map
%}

classdef NoiseMap < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = rf.Sync * rf.Segment * rf.NoiseMapMethod & psy.NoiseMap ...
            & 'noise_map_algorithm in ("STA")'
    end
    
    methods
        
        function plotOri(self)
            % plots the orientation tuning if it exists
            phi = (0:400)/400*(2*pi);
            for siteKey = fetch(rf.Session*rf.Site & 'site>0' & (rf.Scan & self) & (rf.Scan*rf.VonMises))'  % can be used to restrict rf.Scan
                for traceKey = fetch(self*rf.Scan & siteKey, 'scan_idx->rfscan')'
                    
                    subplot 121
                    [pref,base,amp1,amp2,sharp,spatialFreq,tempFreq] = ...
                        fetchn(rf.VonMises*rf.Scan & siteKey & traceKey & 'von_pvalue<0.01' & 'not (spatial_freq=-1 xor temp_freq=-1)', ...
                        'von_pref','von_base','von_amp1','von_amp2','von_sharp','spatial_freq','temp_freq');
                    if ~isempty(pref)
                        von = ne7.rf.VonMises2([base amp1 amp2 sharp pref]);
                        y = von.compute(phi);
                        colors = max(0,min(1,[spatialFreq/0.08 zeros(size(spatialFreq)) tempFreq/4]));
                        lineWidth = 0.25 + 0.75*(spatialFreq==-1 | tempFreq==-1) + 2*(spatialFreq==-1 & tempFreq==-1);
                        plotPolar(phi,y,colors,lineWidth)
                    end
                    
                    subplot 122
                    traceKey.scan_idx = traceKey.rfscan;
                    [rfMap,binSize,degX,degY] = fetch1(self & traceKey,...
                        'rf_map','rf_bin_size','degrees_x','degrees_y');
                    sz = size(rfMap);
                    yaxis = degY/sz(1)*((1:sz(1))-(sz(1)+1)/2);
                    xaxis = degX/sz(2)*((1:sz(2))-(sz(2)+1)/2);
                    taxis = ((1:sz(3))-0.5)*binSize/1000;
                    
                    rfMap = mean(rfMap(:,:,taxis<0.200),3);
                    mx = max(abs(rfMap(:)));
                    imagesc(xaxis,yaxis,rfMap,mx*[-1 1])
                    axis tight image
                    grid on
                    colormap(ne7.vis.doppler)
                    
                    % print
                    f = gcf;
                    f.PaperSize = [2 1]*4.0;
                    f.PaperPosition = [0 0 f.PaperSize];
                    filename = sprintf('~/Google Drive/atlab/RFs/ori/%s-%d/%02d-%03d.png',...
                        fetch1(rf.NoiseMapMethod & traceKey, 'noise_map_algorithm'), ...
                        traceKey.animal_id, traceKey.scan_idx, traceKey.trace_id);
                    suptitle(sprintf('mouse %d; scan %d; cell %d',...
                        traceKey.animal_id, traceKey.scan_idx, traceKey.trace_id))
                    print('-dpng','-r300',filename)
                end
            end
        end
        
        function dump(self)
            for key = self.fetch'
                disp(key)
                map = fetch1(self & key, 'map');
                mx = max(abs(map(:)));
                map = round(map/mx*31.5 + 32.5);
                cmap = ne7.vis.doppler;
                
                for i=1:size(map,3)
                    im = reshape(cmap(map(:,:,i),:),[size(map,1) size(map,2) 3]);
                    f = sprintf('~/dump/%s-%d-%d-%d_%02d.png',...
                        fetch1(rf.NoiseMapMethod & key, 'noise_map_algorithm'), key.animal_id, key.scan_idx, key.trace_id, i);
                    imwrite(im,f)
                end
            end
        end
        
        function plot(self,doSave)
            doSave = nargin<2 || doSave;
            for key = self.fetch'
                disp(key)
                summon1(self & key, 'rf_map','rf_bin_size','degrees_x','degrees_y');
                sz = size(rf_map);
                yaxis = degrees_y/sz(1)*((1:sz(1))-(sz(1)+1)/2);
                xaxis = degrees_x/sz(2)*((1:sz(2))-(sz(2)+1)/2);
                taxis = ((1:sz(3))-0.5)*rf_bin_size/1000;
                
                cla
                rf_map = mean(rf_map(:,:,taxis<0.200),3);
                mx = max(abs(rf_map(:)));
                imagesc(xaxis,yaxis,rf_map,mx*[-1 1])
                axis tight image
                grid on
                title(sprintf('mouse %d; scan %d; cell %d',...
                    key.animal_id, key.scan_idx, key.trace_id))
                
                colormap(ne7.vis.doppler)
                drawnow
                
                if doSave
                    set(gcf,'PaperSize',[4 3],'PaperPosition',[0 0 4 3])
                    f = sprintf('~/Desktop/carfs/%s-%d-%d-%d',...
                        fetch1(rf.NoiseMapMethod & key, 'noise_map_algorithm'), key.animal_id, key.scan_idx, key.trace_id);
                    print('-dpng','-r150',f)
                end
            end
        end
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % temporal binning
            nBins = 20;
            binSize = 0.04;  %s
            
            disp 'loading movies ...'
            caTimes = fetch1(rf.Sync & key,'frame_times');
            dt = median(diff(caTimes));
            trials = rf.Sync*psy.Trial*psy.NoiseMap*psy.NoiseMapLookup;
            trials = trials & key & 'trial_idx between first_trial and last_trial';
            [stimTimes, movie] = trials.fetchn('flip_times', 'cached_movie', 'ORDER BY trial_idx');
            % compute physical dimensions
            sess = fetch(rf.Sync*psy.Session & key,'resolution_x','resolution_y','monitor_distance','monitor_size');
            rect = [sess.resolution_x sess.resolution_y];
            degPerPix = 180/pi*sess.monitor_size*2.54/norm(rect(1:2))/sess.monitor_distance;
            degSize = degPerPix*rect;
            
            disp 'concatenating stimulus movies...'
            stimTimes = cat(2,stimTimes{:});
            movie = double(cat(3,movie{:}))/127-1;
            
            disp 'interpolation...'
            % clip stimulus movie to fit within the calcium recording to avoid extrapolation
            ix = stimTimes > caTimes(1) & stimTimes < caTimes(end) - nBins*binSize;
            movie = movie(:,:,ix);
            stimTimes = stimTimes(ix);
            
            t0 = max(caTimes(1),stimTimes(1)+(nBins-1)*binSize)+0.1;   % start time for calcium traces
            t1 = t0-(nBins-1)*binSize;                                 % start time for stimulus
            t2 = min(caTimes(end),stimTimes(end));                     % end time for both
            movie = permute(interp1(stimTimes',permute(movie,[3 1 2]),(t1:binSize:t2)','linear'),[2 3 1]);
            
            algo = fetch1(rf.NoiseMapMethod & key, 'noise_map_algorithm');
            if strcmp(algo,'STPCA')
                disp 'whitening movie...'
                sz = size(movie);
                movie = reshape(movie,sz(1),sz(2),[]);
                movie = fftn(movie);
                r = sqrt(mean(abs(movie).^2,3));
                r = max(r,0.01*max(r(:)));
                r = min(5,max(r(:))./r(:,1));
                movie = real(ifftn(bsxfun(@times,movie,r)));
                movie = reshape(movie,sz(1)*sz(2),[]);
            end
            
            disp 'computing RF...'
            [traces, traceKey] = fetchn(rf.Trace & key, 'ca_trace');
            for iTrace=1:length(traces)
                fprintf('trace %d\n', traceKey(iTrace).trace_id)
                tuple = dj.struct.join(key,traceKey(iTrace));
                
                % highpass filter and deconvolve
                cutoff = 0.03;
                k = hamming(round(1/dt/cutoff)*2+1);
                k = k/sum(k);
                trace = double(traces{iTrace});
                trace = (trace-ne7.dsp.convmirr(double(trace),k))/mean(trace);
                trace = fast_oopsi(trace,struct('dt',dt),struct('lambda',0.3));
                
                % interpolate to common time bins
                k = hamming(2*round(binSize/dt)+1); k = k/sum(k);  % interpolation kernel
                trace = ne7.dsp.convmirr(trace,k);
                trace = interp1(caTimes,trace,(t0:binSize:t2)','linear');
                trace = trace/sum(trace);
                
                disp 'computing RF...'
                switch algo
                    case 'linear'
                        error 'not implemented yet...'
                    case 'STA'                        
                        % decorrelate spike train
                        sz = size(movie);
                        map = reshape(conv2(fliplr(reshape(movie,sz(1)*sz(2),sz(3))),trace','valid'),sz(1),sz(2),[]);
                        
                    case 'STPCA'
                        % resample movie and stimulus to synchronous times
                        % with the RF binwidth
                        
                        w = zeros(size(movie,1),nBins)+.001;  % largest eigenvector to be computed
                        w = w/norm(w(:));
                        step = 1;
                        for iter=1:16
                            dw = 0;
                            L = 0;
                            fprintf('Iter %d:', iter);
                            for iTime = find(trace>std(trace))'
                                d = movie(:,iTime+(nBins-1:-1:0));
                                y = w(:)'*d(:);
                                dw = dw + trace(iTime)*y*(d-y*w);
                                L = L + trace(iTime)*sum((d(:)-y*w(:)).^2);
                            end
                            fprintf(' Loss=%g',L)
                            dw = dw/norm(dw(:));
                            
                            % line search
                            step = step*1.05;
                            while loss(trace,movie,w+step*dw,nBins)>L
                                step = step*0.5;
                            end
                            w = w + (step*0.5)*dw;   % backtrack a little to be conservative
                            w = w/norm(w(:));
                            fprintf \n
                        end
                        map = reshape(w,sz(1),sz(2),nBins);
                end
                
                disp 'saving..'
                
                tuple.rf_nbins = nBins;
                tuple.rf_bin_size = binSize*1000;
                tuple.degrees_x = degSize(1);
                tuple.degrees_y = degSize(2);
                tuple.rf_map = single(map);
                
                imagesc(map(:,:,2),[-1 1]*0.05), axis image
                colormap(ne7.vis.doppler)
                drawnow
                self.insert(tuple)
            end
            disp done
        end
    end
end


function L = loss(trace,movie,w,nBins)
L = 0;
w = w/norm(w(:));
for iTime=find(trace>std(trace))'
    d = movie(:,iTime+(nBins-1:-1:0));
    y = w(:)'*d(:);
    L = L + trace(iTime)*sum((d(:)-y*w(:)).^2);
end
fprintf(' loss=%g',L)
end


function plotPolar(phi,y,colors,lineWidth)
mx = max(abs(y(:)));
plot(sin(phi)*mx,cos(phi)*mx,'k:','LineWidth',0.25)
hold on
plot(sin(phi)*mx/2,cos(phi)*mx/2,'k:','LineWidth',0.25)
plot([0 0],[-mx mx],'Color',[1 1 1]*0.7,'LineWidth',0.25)
plot([-mx mx],[0 0],'Color',[1 1 1]*0.7,'LineWidth',0.25)
for i=1:size(y,1)
    plot(sin(phi).*y(i,:),cos(phi).*y(i,:),'LineWidth',lineWidth(i),'Color',colors(i,:))
end
axis equal
axis([-mx mx -mx mx])
axis off
hold off
end