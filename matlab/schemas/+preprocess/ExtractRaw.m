%{
preprocess.ExtractRaw (imported) # pre-processing of a twp-photon scan
-> preprocess.Prepare
-> preprocess.Method
---
%}


classdef ExtractRaw < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = preprocess.Prepare*preprocess.Method & (...
            preprocess.PrepareAod*preprocess.MethodAod | ...
            preprocess.PrepareGalvo*preprocess.MethodGalvo)
        tau = 4;
        p = 2;
        max_iter = 2;
        max_neurons = 0;
        nmf_options = 0;
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            %% extract masks and traces for Galvo Scanner
            if count(preprocess.PrepareGalvo & key)
                segmentation_method = fetch1(preprocess.Method*preprocess.MethodGalvo & key, 'segmentation');
                switch segmentation_method
                    case 'nmf'
                        [d2, d1, nslices] = fetch1(preprocess.PrepareGalvo & key, 'px_width', 'px_height','nslices');
                        self.set_nmf_parameters(key);

                        self.insert(key);
                        channel = 1;  % TODO: change to more flexible choice
                        trace_id = 1;
                        for islice = 1:nslices
                            Y = squeeze(self.load_galvo_scan(key, islice, channel));
                            [A,S] = self.run_nmf(Y);
                            traces = A'*reshape(Y,d1*d2,size(Y,3));
                            self.insert_traces(key, traces, channel, trace_id);
                            self.insert_segmentation(key, A, islice, channel, trace_id);
                            trace_id = self.insert_spikes(key, S, channel,trace_id);
                            
                        end
                    otherwise
                        disp(['Not performing ' segmentation_method ' segmentation']);
                        return
                end
                %% extract AOD
            elseif count(preprocess.PrepareAod & key)
                error('AOD trace extraction not implemented.')
            else
                error('Cannot match scan to neither Galvo nor AOD.')
            end
            % 			self.insert(key)
        end
        
        
        function set_nmf_parameters(self, key)
            
            switch fetch1(experiment.Scan & key, 'aim')
                case {'unset', 'functional: somas'}
                    
                    [d2, d1, um_width, um_height] = fetch1(preprocess.PrepareGalvo & key, 'px_width', 'px_height', 'um_width', 'um_height');
                    self.tau = 4;
                    self.p = 2;
                    neuron_density = 800; % neuron per mm^2 slice
                    self.max_iter = 2;
                    downsample_to = 4;
                    self.nmf_options = CNMFSetParms(...
                        'd1',d1,'d2',d2,...                         % dimensions of datasets
                        'search_method','ellipse','dist',3,...      % search locations when updating spatial components
                        'deconv_method','constrained_foopsi',...    % activity deconvolution method
                        'temporal_iter',2,...                       % number of block-coordinate descent steps
                        'fudge_factor',0.98,...                     % bias correction for AR coefficients
                        'merge_thr',.8,...                          % merging threshold
                        'gSig', self.tau, ...
                        'se',  strel('disk',3,0),...
                        'init_method', 'greedy_corr', ...
                        'tsub', floor(fetch1(preprocess.PrepareGalvo & key, 'fps')/downsample_to) ...
                        );
                    self.max_neurons = round((um_width * um_height)/1000^2 * neuron_density);
                    fprintf('Using max %i neurons\n', self.max_neurons);
                case 'functional: axons'
                    error('Segmentation for axons not implemented yet');
                otherwise
                    error('Segmentation for this aim is not implemented!');
            end
        end
        
        function [A,S] = run_nmf(self, Y)
            %
            % Runs the nonnegative matrix factorization algorithm on Y with configuration cfg.
            %
            
            [d1,d2, T] = size(Y);
            d = d1*d2;
            [P,Y] = preprocess_data(Y,self.p);
           
            
            % fast initialization of spatial components using greedyROI and HALS
            [A,C,~,f,~] = initialize_components(Y,self.max_neurons,self.tau,self.nmf_options,P);  % initialize
            
            % update spatial and temporal components
            Yr = reshape(Y,d,T);
            clear Y;
            
            % update spatial components
            for iter = 1:self.max_iter
                [A,b,C] = update_spatial_components(Yr,C,f,A,P,self.nmf_options);
                [C,f,P,S] = update_temporal_components(Yr,A,b,C,f,P,self.nmf_options);
                
                % merge found components
                [A,C,~,~,P,S] = merge_components(Yr,A,b,C,f,P,S,self.nmf_options);
            end
            [A,~,S,~] = order_ROIs(A,C,S,P);    % order components
        end        
        
    end
    
    methods(Static)
        
        %%------------------------------------------------------------
        function scan = load_galvo_scan(key, islice, ichannel, maxT)
            
            reader = preprocess.getGalvoReader(key);
            
            nframes = fetch1(preprocess.PrepareGalvo & key, 'nframes');
            
            if nargin < 4
                maxT = reader.nframes;
            end
            
            [r,c] = fetch1(preprocess.PrepareGalvo & key, 'px_height', 'px_width');
            
            fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
            key.slice = islice;
            fixMotion = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
            fprintf('Loading slice %d channel %d\n', islice, ichannel)
            scan = zeros(r,c, 1, maxT);
            for iframe = 1:min(nframes, maxT)
                scan(:, :, 1, iframe) = fixMotion(fixRaster(reader(:,:,ichannel, islice, iframe)), iframe);
            end
            
        end
        
        %%
        function trace_id = insert_traces(key, traces, channel, trace_id)
            if nargin < 4
                trace_id = 1;
            end
            
            for itrace = 1:size(traces, 1)
                trace_key = struct(key);
                trace_key.channel = channel;
                trace_key.trace_id = trace_id;
                trace_key.raw_trace = traces(itrace, :);
                insert(preprocess.ExtractRawTrace, trace_key);
                trace_id = trace_id + 1;
            end
            
        end
        
        %%
        function trace_id = insert_segmentation(key, A,  slice, channel, trace_id)
            
            seg_key = struct(key);
            seg_key.slice = slice;
            insert(preprocess.ExtractRawGalvoSegmentation, seg_key);
            
            for itrace = 1:size(A, 2)
                mask_key = struct(seg_key);
                mask_key.channel = channel;
                mask_key.trace_id =  trace_id;
                I = find( A(:,itrace));
                mask_key.mask_pixels = I;
                mask_key.mask_weights = A(I, itrace);
                insert(preprocess.ExtractRawGalvoROI, mask_key);
                trace_id = trace_id + 1;
            end
        end        
        
        %%
        function trace_id = insert_spikes(key, S, channel, trace_id)
            
            for itrace = 1:size(S, 1)
                spike_key = struct(key);
                spike_key.channel = channel;
                spike_key.trace_id = trace_id;
                spike_key.spike_trace = S(itrace, :);
                insert(preprocess.ExtractRawSpikeRate, spike_key);
                trace_id = trace_id + 1;
            end
            
        end               
        
    end
    
    
end