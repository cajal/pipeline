%{
tuning.DotRF (computed) # receptive field from the dot stimulus
-> preprocess.Sync
-> preprocess.Spikes
-> tuning.DotRFMethod
-----
%}

classdef DotRF < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = tuning.DotRFMethod * preprocess.Spikes & ...
            (preprocess.Sync & (vis.ScanConditions & vis.SingleDot))
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % get parameters
            [mon_size, mon_dist] = fetch1(vis.Session &...
                (preprocess.Spikes * preprocess.Sync & vis.SingleDot & key), ...
                'monitor_size','monitor_distance');
            [onset_delay, response_duration, rf_filter, shuffle, sd] = ...
                fetch1(tuning.DotRFMethod & key, ...
                'onset_delay', 'response_duration', 'rf_filter', 'shuffle', 'rf_sd');
            
            % get stimulus conditions
            trials = (preprocess.Sync * vis.Trial * vis.SingleDot ...
                    & key & 'trial_idx between first_trial and last_trial');
                
            conds = fetch(vis.Condition * vis.SingleDot & trials.fetch);
            [locations_x,locations_y] = fetchn(vis.SingleDot & conds, 'dot_x', 'dot_y');
            locations_x = unique(locations_x);
            locations_y = unique(locations_y);
            map_size = [length(locations_x) length(locations_y)];
            deg2dot = 2*atand((mon_size/2)/(mon_dist/2.54))\norm(map_size);
            
            % get traces and frame times
            [traces, frame_times, trace_keys] = pipetools.getAdjustedSpikes(key);
            xm = min([length(frame_times) length(traces)]);
            frame_times = frame_times(1:xm);
            traces = traces(1:xm,:);
            
            % get responses for each trial
            trials = trials.fetch('cond_idx', 'dot_x', 'dot_y', 'flip_times');
            response = nan(length(trials),size(traces,2));
            index = nan(length(trials),1);
            for itrial = 1:length(trials)
                trial = trials(itrial);
                frame_rel = frame_times<trial.flip_times(1)+response_duration/1000 +onset_delay/1000 ...
                    & frame_times>trial.flip_times(1)+onset_delay/1000;
                index(itrial) = sub2ind(map_size, find(locations_x==trial.dot_x), find(locations_y==trial.dot_y));
                response(itrial,:) = nanmean(traces(frame_rel,:));
            end
            
            % shuffle trials for bootstrap computation
            trial_length = size(response,1);
            sfl_resp = nan(size(response,1),size(response,2),shuffle);
            for icell = 1:size(response,2)
                sfl_resp(:,icell,1) = response(randperm(trial_length),icell);
                for i = 2:shuffle
                    sfl_resp(:,icell,i) = sfl_resp(randperm(trial_length),icell,i-1);
                end
            end
            sfl_resp_p = nan(size(response,1),shuffle);
            resp_p = mean(response,2);
            sfl_resp_p(:,1) = resp_p(randperm(trial_length));
            for i = 2:shuffle
                sfl_resp_p(:,i) = sfl_resp_p(randperm(trial_length),i-1);
            end
            
            % average across trials
            response_map = nan(map_size(1),map_size(2),size(response,2));
            sfl_response_map = nan(map_size(1),map_size(2),size(response,2),shuffle);
            sfl_resp_map_p = nan(map_size(1),map_size(2),shuffle);
            for iloc = unique(index)'
                [x, y] = ind2sub(map_size, iloc);
                response_map(x,y,:) = nanmean(response(index==iloc,:));
                sfl_response_map(x,y,:,:) = nanmean(sfl_resp(index==iloc,:,:));
                sfl_resp_map_p(x,y,:) = nanmean(sfl_resp_p(index==iloc,:));
            end
            
            % insert
            self.insert(key)
            
            % compute and insert pop rf
            key.response_map = mean(response_map,3);
            key.gauss_fit = self.fitGauss(key.response_map, deg2dot, rf_filter);
            key.snr = self.rfSNR(key.gauss_fit, key.response_map,sd);
            key.p_value = mean(key.snr<squeeze(self.rfSNR(key.gauss_fit, sfl_resp_map_p,sd)));
            key.center_y = (key.gauss_fit(1) - map_size(2)/2 - 0.5) / map_size(1);
            key.center_x = (key.gauss_fit(2) - map_size(1)/2 - 0.5) / map_size(1);
            insert(tuning.DotRFMapPop,key);
            
            % compute and insert cell rfs
            for itrace = 1:length(trace_keys)
                tuple = rmfield(trace_keys(itrace),'slice');
                tuple.rf_method = key.rf_method;
                tuple.response_map = response_map(:,:,itrace);
                tuple.gauss_fit = self.fitGauss(tuple.response_map, deg2dot, rf_filter);
                tuple.snr = self.rfSNR(tuple.gauss_fit, tuple.response_map,sd);
                tuple.p_value = mean(tuple.snr<squeeze(...
                    self.rfSNR(tuple.gauss_fit, sfl_response_map(:,:,itrace,:),sd)));
                tuple.center_y = (tuple.gauss_fit(1) - map_size(2)/2 - 0.5) / map_size(1);
                tuple.center_x = (tuple.gauss_fit(2) - map_size(1)/2 - 0.5) / map_size(1);
                insert(tuning.DotRFMap,tuple);
            end
        end
        
        function par = fitGauss(self,z,deg2dot,gaussW)
            
            % apply smoothing
            w = window(@gausswin,round(gaussW*deg2dot));
            w = w * w';
            w = w / sum(w(:));
            z = imfilter(z,w,'circular');
            sz = size(z);
            
            % initialize fit parameters
            [x,y] = meshgrid(1:size(z,2),1:size(z,1));
            x = [x(:) y(:)]';  z = z(:);
            [amp, i] = max(z);
            base = prctile(z,10);
            par = [x(1,i) x(2,i) 1 1 0 amp-base base]';
            
            % fit a 2D gaussian
            lb = [0 0 0.5 0.5 -0.5 -inf -inf];
            ub = [sz(2) sz(1) 2 2 0.5 inf inf];
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
        
        function SNR = rfSNR(self,par,z,sd) % compute RF SNR
            sz = size(z);
            z = reshape(z,[],size(z,3),size(z,4));
            mu = par(1:2)';
            CC=diag(par(3:4)); CC(1,2)=par(5); CC(2,1)=par(5);
            [x,y] = meshgrid(1:sz(2),1:sz(1));
            X=[x(:) y(:)];
            X = bsxfun(@minus, X, mu);
            d = sum((X /CC) .* X, 2);
            noise = var(z(d > sd,:,:),[],1);
            sig = nanvar(z(d < sd,:,:),[],1);
            SNR = sig ./ noise;
            if isnan(SNR);SNR = 0;end
        end
    end
    
    methods
        
        function plot(self, gaussfit, map)
            
            params.color = [0 0 1];
            params.line = '-';
            params.linewidth = 1;
            params.npts = 50;
            
            % plot response map
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
            sd = fetch1(self * tuning.DotRFMethod,'rf_sd');
            tt=linspace(0,2*pi,params.npts)';
            x = cos(tt); y=sin(tt);
            ap = [x(:) y(:)]';
            [v,d]=eig(C);
            d(d<0) = 0;
            d = sd * sqrt(d); % convert variance to sdwidth*sd
            bp = (v*d*ap) + repmat(m, 1, size(ap,2));
            
            % plot rf
            plot(bp(2,:), bp(1,:),params.line,'Color',params.color,'linewidth',params.linewidth);
            plot(m(2),m(1),'*r','markersize',5)
            
            % adjust labels as a fraction of x
            yrange = size(map,2)/2/size(map,1);
            roundall = @(x) round(x*100)/100;
            set(gca,'xtick',linspace(0.5,size(map,1)+0.5,10),'xticklabel',roundall(linspace(-0.5,0.5,10)))
            set(gca,'ytick',linspace(0.5,size(map,2)+0.5,10),'yticklabel',roundall(linspace(-yrange,yrange,10)))
            grid on
            axis image
        end
    end
end