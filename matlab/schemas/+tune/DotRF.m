%{
# 
-> stimulus.Sync
-> fuse.ScanDone
-> tune.DotRFMethod
%}

classdef DotRF < dj.Computed
    
    properties
        popRel = tune.DotRFMethod * fuse.ScanDone * stimulus.Sync & (stimulus.Trial & stimulus.SingleDot)
    end
    
	methods(Access=protected)

		function makeTuples(self, key)
            % get parameters
            [mon_size, mon_dist] = fetch1(experiment.DisplayGeometry & ...
                (fuse.Activity * stimulus.Sync & stimulus.SingleDot & key), ...
                'monitor_size','monitor_distance');
            [onset_delay, response_duration, rf_filter, shuffle, sd, snr_method] = ...
                fetch1(tune.DotRFMethod & key, ...
                'onset_delay', 'response_duration', 'rf_filter', 'shuffle', 'rf_sd','snr_method');
            
            % get stimulus conditions
            trials = (stimulus.Sync * stimulus.Trial * stimulus.SingleDot & key);
            conds = fetch(stimulus.Condition * stimulus.SingleDot & proj(trials));
            [locations_x,locations_y,levels] = fetchn(stimulus.SingleDot & conds, 'dot_x', 'dot_y', 'dot_level');
            locations_x = unique(locations_x);
            locations_y = unique(locations_y);
            levels = unique(levels); if length(levels)==1; levels(2)= nan;end
            map_size = [length(locations_x) length(locations_y)];
            deg2dot = 2*atand((mon_size/2)/(mon_dist/2.54))\norm(map_size);
            
            % get traces and frame times
            [traces, frame_times, trace_keys] = getAdjustedSpikes(fuse.ActivityTrace & key);
            
            % get responses for each trial
            trials = trials.fetch('dot_level', 'condition_hash', 'dot_x', 'dot_y', 'flip_times');
            response = nan(length(trials),length(levels),size(traces,2));
            index = nan(length(trials),1);
            for itrial = 1:length(trials)
                trial = trials(itrial);
                frame_rel = frame_times<trial.flip_times(1)+response_duration/1000 +onset_delay/1000 ...
                    & frame_times>trial.flip_times(1)+onset_delay/1000;
                index(itrial) = sub2ind(map_size, find(locations_x==trial.dot_x), find(locations_y==trial.dot_y));
                response(itrial,(trial.dot_level==levels(2))+1,:) = single(nanmean(traces(frame_rel,:),1));
            end
            
            % clean up memory and shrink variable size.
            clear traces frame_times trials conds;
            
            % shuffle trials
            sfl_resp_p = nan(size(response,1),size(response,2),shuffle);
            trial_length = size(response, 1);
            resp_p = mean(response,3);
            sfl_resp_p(:,:,1) = resp_p(randperm(trial_length),:);
            for i = 2:shuffle
                sfl_resp_p(:,:,i) = sfl_resp_p(randperm(trial_length),:,i-1);
            end
                        
            % average across trials
            response_map = nan(map_size(1),map_size(2),length(levels),size(response,3));
            sfl_resp_map_p = nan(map_size(1),map_size(2),length(levels),shuffle);
            for iloc = unique(index)'
                [x, y] = ind2sub(map_size, iloc);
                response_map(x,y,:,:) = nanmean(response(index==iloc,:,:),1);
                sfl_resp_map_p(x,y,:,:) = nanmean(sfl_resp_p(index==iloc,:,:),1);
            end
            
            % insert
            self.insert(key)
            
            % compute and insert pop rf
            key.response_map = mean(response_map,4);  % (off, on) responses in third dimension.
            key.gauss_fit = self.fitGauss(nanmax(key.response_map,[],3), deg2dot, rf_filter);
            key.snr = self.rfSNR(key.gauss_fit, nanmax(key.response_map,[],3),sd, snr_method);
            key.p_value = mean(key.snr<squeeze(self.rfSNR(key.gauss_fit, max(sfl_resp_map_p,[],3),sd, snr_method)));
            key.center_y = (key.gauss_fit(1) - map_size(2)/2 - 0.5) / map_size(1);
            key.center_x = (key.gauss_fit(2) - map_size(1)/2 - 0.5) / map_size(1);
            insert(tune.DotRFMapPop,key);
           
            % compute and insert cell rfs
            tuples=[];
            parfor itrace = 1:length(trace_keys)
                tuple = trace_keys(itrace);
                tuple.rf_method = key.rf_method;
                tuple.response_map = response_map(:,:,:,itrace);
                tuple.gauss_fit = self.fitGauss(max(tuple.response_map,[],3), deg2dot, rf_filter);
                tuple.snr = self.rfSNR(tuple.gauss_fit, max(tuple.response_map,[],3),sd,snr_method);
                
                % shuffle across trials
                sfl = nan(size(response,1),size(response,2),shuffle);
                sfl(:,:,1) = response(randperm(trial_length),:,itrace);
                for i = 2:shuffle
                    sfl(:,:,i) = sfl(randperm(trial_length),:,i-1);
                end
                
                sfl_response_map = nan(map_size(1),map_size(2),length(levels),shuffle);
                for iloc = unique(index)'
                    [x,y] = ind2sub(map_size, iloc);
                    sfl_response_map(x,y,:,:) = nanmean(sfl(index==iloc,:,:),1);
                end
                
                tuple.p_value = mean(tuple.snr<squeeze(...
                    self.rfSNR(tuple.gauss_fit, max(sfl_response_map(:,:,:,:),[],3),sd, snr_method)));
                tuple.center_y = ((tuple.gauss_fit(1) / map_size(2))-0.5)*map_size(2)/map_size(1);
                tuple.center_x = (tuple.gauss_fit(2) / map_size(1))-0.5;

                tuples = [tuples; tuple];
            end
            
            for itrace = 1:length(trace_keys)
                tuple = tuples(itrace);
                insert(tune.DotRFMap,tuple);
            end
            
        end
        
        function par = fitGauss(~,z,deg2dot,gaussW)
            
            % apply smoothing
%             w = window(@gausswin,round(gaussW*deg2dot));
%             w = w * w';
%             w = w / sum(w(:));
%             z = imfilter(z,w,'circular');
            sz = size(z);
            
            % initialize fit parameters
            [x,y] = meshgrid(1:size(z,2),1:size(z,1));
            x = [x(:) y(:)]';  z = z(:);
            [amp, i] = max(z);
            base = prctile(z,10);
            par = [x(1,i) x(2,i) 1 1 0 amp-base base]';  
            % intial guess of parameters for gaussian fit
            %[centerx,centery, sd1,sd2,offdiag, amp, base].
            
            % fit a 2D gaussian
            lb = [0 0 0.5 0.5 -0.5 -inf -inf];
            ub = [sz(2) sz(1) 4 4 0.5 inf inf];
            opt = optimset('Display','off','MaxFunEvals',1e20,'MaxIter',1e3);
            [par, ~] = lsqcurvefit(@Gauss,par,x,z',lb,ub,opt);
            
            % 2D gauss function
            function z = Gauss(par,x)
                m = par(1:2);
                C = diag(par(3:4));
                cc = par(5) * sqrt(prod(par(3:4)));
                C(1,2) = cc;
                C(2,1) = cc;
                xx = bsxfun(@minus,x,m);
                z = exp(-.5*sum(xx.*(inv(C)*xx),1)) * par(6) + par(7);
            end
        end
        
        function SNR = rfSNR(~,par,z,sd, method) % compute RF SNR
            sz = size(z);
            z = reshape(z,[],size(z,3),size(z,4));
            mu = par(1:2)';
            CC=diag(par(3:4)); CC(1,2)=par(5); CC(2,1)=par(5);
            [x,y] = meshgrid(1:sz(2),1:sz(1));
            X=[x(:) y(:)];
            X = bsxfun(@minus, X, mu);
            d = sum((X /CC) .* X, 2);
            noise = z(d > sd, :,:);
            sig = z(d < sd, :,:);
            switch method
                case 'var'
                    SNR = nanvar(sig,[],1) ./ nanvar(noise,[],1);
                case 'zscore'
                    SNR = (nanmean(sig,1) - nanmean(noise,1)) ./ sqrt(nanvar(noise,[],1));
                case 'dprime'
                    mu1 = nanmean(sig,1);
                    mu2 = nanmean(noise, 1);
                    var1 = nanvar(sig,[],1);
                    var2 = nanvar(noise,[],1);
                    SNR = (mu1 - mu2) ./ sqrt(0.5*(var1 + var2));
            end
                
            if isnan(SNR);SNR = 0;end
        end
    end
    
    methods
        
        function plot(self, gaussfit, map, varargin)
            
            params.color = [0 0 1];
            params.line = '-';
            params.linewidth = 1;
            params.npts = 50;
            params.scale = 1;
            
            params = ne7.mat.getParams(params, varargin);
            
            % plot response map
            if params.scale~=1
                map = imresize(map,params.scale);
            end
            imagesc(map');
            hold on
            colormap gray
            
            % get fit parameters
            m=gaussfit(1:2);
            C=diag(gaussfit(3:4));
            cc=gaussfit(5)*sqrt(prod(gaussfit(3:4)));
            C(1,2)=cc;
            C(2,1)=cc;
            
            % estimate rf boundaries
            sd = fetchn(self * tune.DotRFMethod,'rf_sd');
            sd = unique(sd);
            tt=linspace(0,2*pi,params.npts)';
            x = cos(tt); y=sin(tt);
            ap = [x(:) y(:)]';
            [v,d]=eig(C);
            d(d<0) = 0;
            d = sd * sqrt(d); % convert variance to sdwidth*sd
            bp = (v*d*ap) + repmat(m, 1, size(ap,2));
            
            % plot rf
            plot(bp(2,:)*params.scale, bp(1,:)*params.scale,params.line,'Color',params.color,'linewidth',params.linewidth);
            plot(m(2)*params.scale,m(1)*params.scale,'*r','markersize',5)
            
            % adjust labels as a fraction of x
            steps = 5;
            x_idx = loc2deg(self,linspace(-0.5,0.5,steps),1);
            y_idx = loc2deg(self,linspace(-0.33,0.33,steps),1);
            set(gca,'xtick',linspace(1,size(map,1),steps),'xticklabel',round(x_idx))
            set(gca,'ytick',linspace(1,size(map,2),steps),'yticklabel',round(y_idx))
            grid on
            axis image
        end
        
        function deg = loc2deg(obj,loc,flat_corrected)
            [aspect, distance, diag_size] = fetch1(experiment.DisplayGeometry & obj,'monitor_aspect','monitor_distance','monitor_size');
            x_size = sind(atand(aspect))*diag_size;
            if nargin>1 && flat_corrected
                x2deg = @(xx) atand(x_size*xx/distance);
                deg = x2deg(loc);
            else
                max_deg = atand(x_size/2/distance)*2;
                deg = loc*max_deg;
            end
        end

    end

end