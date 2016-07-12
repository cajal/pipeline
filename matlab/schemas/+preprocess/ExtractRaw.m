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
        mask_range = 0;
        batchsize = 0;
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            %% extract masks and traces for Galvo Scanner
            if count(preprocess.PrepareGalvo & key)
                segmentation_method = fetch1(preprocess.Method*preprocess.MethodGalvo & key, 'segmentation');
                switch segmentation_method
                    case 'nmf'
                        fprintf('NMF segmentation for Gavlo scan\n')
                        [d2, d1, nslices, nframes] = fetch1(preprocess.PrepareGalvo & key, 'px_width', 'px_height','nslices','nframes');
                        self.set_nmf_parameters(key);
                        
                        channel = 1;  % TODO: change to more flexible choice
                        trace_id = 1;
                        fprintf('\tInitializing reader\n');
                        reader = preprocess.getGalvoReader(key);
                        
                        self.insert(key);
                        for islice = 1:nslices
                            fprintf('\tLoading data %i:%i for slice %i\n', self.mask_range(1), self.mask_range(end), ...
                                islice);
                            Y = squeeze(self.load_galvo_scan(key, islice, channel, self.mask_range, reader));
                            
                            fprintf('\tInferring masks\n');
                            try 
                                [A, b] = self.run_nmf(Y);
                            catch ME
                                if strcmp(ME.identifier,'MATLAB:imresize:expectedNonempty')
                                    fprintf('\tCaught non-empty exception. No neuron found!\n');
                                    return 
                                end
                            end
                            fprintf('\tMasks inferred successfully. Inferring spikes\n');
                            traces = zeros(size(A,2), nframes);
                            S = 0*traces;
                            for start = 1:self.batchsize:nframes
                                idx = start:min(nframes, start + self.batchsize - 1);
                                fprintf('\t\tInferring spikes for frames %i:%i\n', idx(1), idx(end));
                                Y = squeeze(self.load_galvo_scan(key, islice, channel, idx));
                                traces(:, idx) = A'*reshape(Y,d1*d2,size(Y,3));
                                S(:,idx) = self.nmf_spikes(A, b, Y);
                            end
                            S(isnan(traces)) = nan;
                            
                            fprintf('\tInserting masks, traces, and spikes\n');
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
            
            fprintf('\tSetting NMF parameters for functional scan on somas\n');
            mask_chunk = 10*60; % chunk size in seconds
            self.batchsize = 10000; % batch size for temporal processing
            [d2, d1, um_width, um_height, nframes] = fetch1(preprocess.PrepareGalvo & key, 'px_width', 'px_height', 'um_width', 'um_height','nframes');
            self.tau = 4;
            self.p = 2;
            neuron_density = 800; % neuron per mm^2 slice
            self.max_iter = 2;
            downsample_to = 4;
            fps = fetch1(preprocess.PrepareGalvo & key, 'fps');
            self.nmf_options = CNMFSetParms(...
                'd1',d1,'d2',d2,...                         % dimensions of datasets
                'search_method','ellipse','dist',3,...      % search locations when updating spatial components
                'deconv_method','constrained_foopsi',...    % activity deconvolution method
                'temporal_iter',2,...                       % number of block-coordinate descent steps
                'fudge_factor',0.98,...                     % bias correction for AR coefficients
                'merge_thr',.8,...                          % merging threshold
                'gSig', self.tau, ... % TODO: possibly update gsig to adapt to neuron size in FOV
                'se',  strel('disk',3,0),...
                'init_method', 'greedy_corr', ...
                'tsub', floor(fps/downsample_to) ...
                );
            self.max_neurons = round((um_width * um_height)/1000^2 * neuron_density);
            mc = min(ceil(mask_chunk * fps), nframes);
            tmp = ceil((nframes - mc)/2);
            if tmp < 1
                self.mask_range = 1:nframes;
            else
                self.mask_range = tmp:tmp+mc-1;
            end
            fprintf('\tUsing max %i neurons and %i frames to infer masks\n', self.max_neurons, ...
                length(self.mask_range));
            fprintf('\tBatchsize is %i\n', self.batchsize);
        end
        
        
        function S = nmf_spikes(self, A, b, Y)
            %P = struct('p',1);
            [d1,d2, T] = size(Y);
            d = d1*d2;
            [P,Y] = preprocess_data(Y,self.p);
            % update spatial and temporal components
            tsub = self.nmf_options.tsub;
            self.nmf_options.tsub = 1;
            [C,f,P,S] = update_temporal_components(reshape(Y,d,T),A,b,[],[],P,self.nmf_options);
            self.nmf_options.tsub = tsub;
        end
        
        function [A, b, P] = run_nmf(self, Y)
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
            [A,C,S,P] = order_ROIs(A,C,S,P);    % order components
        end
        
    end
    
    methods(Static)
        
        %%------------------------------------------------------------
        function scan = load_galvo_scan(key, islice, ichannel, frames, reader)
            
            if nargin < 5
                reader = preprocess.getGalvoReader(key);
            end
            
            %             nframes = fetch1(preprocess.PrepareGalvo & key, 'nframes');
            
            if nargin < 4
                frames = 1:reader.nframes;
            end
            
            [r,c] = fetch1(preprocess.PrepareGalvo & key, 'px_height', 'px_width');
            
            fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
            key.slice = islice;
            fixMotion = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
            fprintf('\tLoading slice %d channel %d\n', islice, ichannel)
            scan = zeros(r,c, 1, length(frames));
            N = length(frames);
            
            for iframe = frames
                if mod(iframe, 1000) == 0
                    fprintf('\r\t\tloading frame %i (%i/%i)', iframe, iframe - frames(1) + 1, N);
                end
                scan(:, :, 1, iframe - frames(1) + 1) = fixMotion(fixRaster(reader(:,:,ichannel, islice, iframe)), iframe);
            end
            fprintf('\n');
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