%{
preprocess.ExtractRaw (imported) # pre-processing of a twp-photon scan
-> preprocess.Prepare
-> preprocess.Method
---
%}


classdef ExtractRaw < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = preprocess.Prepare*preprocess.Method ...
            & ((preprocess.PrepareGalvo*preprocess.MethodGalvo - 'segmentation="manual"') | ... % do all automatic methods
            preprocess.PrepareGalvo*preprocess.MethodGalvo*preprocess.ManualSegment | ... % but only manual for manual segmented scans
            preprocess.PrepareAod*preprocess.MethodAod) ...% or AOD methods for AOD scans
            - (experiment.SessionTargetStructure & 'compartment="axon"'); % but no axons at the moment
        tau = 4;
        p = 2;
        max_iter = 2;
        max_neurons = 0;
        nmf_options = 0;
        mask_range = 0;
        batchsize = 5000;
        min_block_size = 0;
        nan_tol = 0;
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            
            % extract masks and traces for Galvo Scanner
            if count(preprocess.PrepareGalvo & key)
                segmentation_method = fetch1(preprocess.Method*preprocess.MethodGalvo & key, 'segmentation');
                switch segmentation_method
                    case 'nmf'
                        fprintf('NMF segmentation for Galvo scan\n')
                        [d2, d1, nslices, nframes] = fetch1(preprocess.PrepareGalvo & key, 'px_width', 'px_height','nslices','nframes');
                        self.set_nmf_parameters(key);
                        
                        fprintf('\tInitializing reader\n');
                        [loader, channel] = preprocess.create_loader(key);
                        
                        self.insert(key);
                        trace_id = 1;
                        
                        for islice = 1:nslices
                            fprintf('\tLoading data %i:%i for slice %i\n', self.mask_range(1), self.mask_range(end), ...
                                islice);
                            
                            
                            Y = loader(islice, self.mask_range);
                            notnan = preprocess.getblocks(squeeze(any(any(~isnan(Y), 1),2)), round(length(self.mask_range)/2), self.nan_tol);
                            
                            if length(notnan) ~= 1
                                error('Too many NaNs in frames for mask inference');
                            end
                            
                            fprintf('\tInferring masks\n');
                            try
                                [A, b] = self.run_nmf(Y(:,:,notnan{1}));
                            catch ME
                                if strcmp(ME.identifier,'MATLAB:imresize:expectedNonempty')
                                    fprintf('\tCaught non-empty exception. No neuron found! Continuing to next slice. \n');
                                    continue
                                else
                                    throw(ME)
                                end
                            end
                            
                            fprintf('\tMasks inferred successfully. Inferring spikes\n');
                            traces = nan(size(A,2), nframes);
                            spike_traces = 0*traces*NaN;
                            for start = 1:self.batchsize:nframes
                                batch = start:min(nframes, start + self.batchsize - 1);
                                
                                fprintf('\t\tInferring spikes for frames %i:%i\n', batch(1), batch(end));
                                Y = loader(islice, batch);
                                notnan = preprocess.getblocks(squeeze(any(any(~isnan(Y), 1),2)), self.min_block_size, self.nan_tol);
                                
                                for i = 1:length(notnan)
                                    blk = notnan{i};
                                    traces(:, batch(blk)) = A'*reshape(Y(:,:,blk),d1*d2,length(blk));
                                    spike_traces(:,batch(blk)) = self.nmf_spikes(A, b, Y(:,:,blk));
                                end
                            end
                            
                            fprintf('\tInserting masks, traces, and spikes\n');
                            self.insert_traces(key, traces, channel, trace_id);
                            self.insert_segmentation(key, A, islice, channel, trace_id);
                            trace_id = self.insert_spikes(key, spike_traces, channel,trace_id);
                            
                        end
                    case 'manual'
                        [d2, d1, nslices, nframes] = fetch1(preprocess.PrepareGalvo & key, 'px_width', 'px_height','nslices','nframes');
                        [loader, channel] = preprocess.create_loader(key,0);
                        self.insert(key)
                        trace_id = ones(length(channel),1);
                        for islice = 1:nslices
                            key.slice = islice;
                            insert(preprocess.ExtractRawGalvoSegmentation, key);
                            mask_image = fetch1(preprocess.ManualSegment & key, 'mask');
                            if max(mask_image(:))==1 % adds backwards compatibility with drawCells
                                mask_image = bwlabel(mask_image, 4);
                            end
                            regions = regionprops(mask_image,'PixelIdxList');
                            mask_pixels = {regions(:).PixelIdxList};
                            trace_key = rmfield(key,'slice');
                            traces = zeros(length(mask_pixels), length(channel), nframes);
                            
                            for start = 1:self.batchsize:nframes
                                batch = start:min(nframes, start + self.batchsize - 1);
                                fprintf('\tComputing traces for frames %i:%i\n', batch(1), batch(end));
                                
                                Y = loader(islice, batch);
                                for imask = 1:length(mask_pixels)
                                    for ichannel = 1:length(channel)
                                        [x,y] = ind2sub([d1 d2],mask_pixels{imask});
                                        traces(imask, ichannel, batch) = squeeze(nanmean(nanmean(Y(x,y,ichannel, :))));
                                    end
                                end
                                
                            end
                            
                            for imask = 1:length(mask_pixels)
                                for ichannel = 1:length(channel)
                                    trace_key.channel = channel(ichannel);
                                    trace_key.trace_id = trace_id(ichannel);
                                    trace_key.raw_trace = squeeze(single(traces(imask, ichannel, :)));
                                    insert(preprocess.ExtractRawTrace, trace_key);
                                    
                                    mask_key = key;
                                    mask_key.channel = channel(ichannel);
                                    mask_key.trace_id =  trace_id(ichannel);
                                    mask_key.mask_pixels = mask_pixels{imask};
                                    mask_key.mask_weights = ones(size(mask_pixels{imask}));
                                    insert(preprocess.ExtractRawGalvoROI, mask_key);
                                    trace_id(ichannel) = trace_id(ichannel) + 1;
                                end
                            end
                            
                        end
                    otherwise
                        disp(['Not performing ' segmentation_method ' segmentation']);
                        return
                end
                % extract AOD
            elseif count(preprocess.PrepareAod & key)
                error('AOD trace extraction not implemented.')
            else
                error('Cannot match scan to neither Galvo nor AOD.')
            end
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
            downsample_to = 4; % temporal downsampling to this rate in Hz during mask inference
            fps = fetch1(preprocess.PrepareGalvo & key, 'fps');
            self.min_block_size = round(fps*5); % minimum block size for spike inference
            self.nan_tol = round(fps/10); % maximal stretch of nans to tolerate
            
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
            
            if count(experiment.SessionTargetStructure & key) == 0 || ...
                    strcmp(fetch1(experiment.SessionTargetStructure & key,'compartment'),'soma')
                fprintf('\tSetting parameters for soma segmentation\n');
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
            elseif strcmp(fetch1(experiment.SessionTargetStructure & key,'compartment'),'axon')
                error('Axons are currently not processed. ');
                fprintf('\tSetting parameters for axon segmentation\n');
                self.nmf_options = CNMFSetParms(...
                    'd1',d1,'d2',d2,...                         % dimensions of datasets
                    'search_method','dilate',...      % search locations when updating spatial components
                    'deconv_method','constrained_foopsi',...    % activity deconvolution method
                    'temporal_iter',2,...                       % number of block-coordinate descent steps
                    'fudge_factor',0.98,...                     % bias correction for AR coefficients
                    'merge_thr',.8,...                          % merging threshold
                    'init_method', 'HALS', ...
                    'tsub', floor(fps/downsample_to) ...
                    );
            end
        end
        
        
        function spike_traces = nmf_spikes(self, A, b, Y)
            %P = struct('p',1);
            [d1,d2, T] = size(Y);
            d = d1*d2;
            [P,Y] = preprocess_data(Y,self.p);
            % update spatial and temporal components
            tsub = self.nmf_options.tsub;
            self.nmf_options.tsub = 1;
            [C,f,P,spike_traces] = update_temporal_components(reshape(Y,d,T),A,b,[],[],P,self.nmf_options);
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
        
        
        function trace_id = insert_traces(key, traces, channel, trace_id)
            if nargin < 4
                trace_id = 1;
            end
            
            for itrace = 1:size(traces, 1)
                trace_key = struct(key);
                trace_key.channel = channel;
                trace_key.trace_id = trace_id;
                trace_key.raw_trace = single(traces(itrace, :))'; % convert to single and make a column vector, see issue #115
                insert(preprocess.ExtractRawTrace, trace_key);
                trace_id = trace_id + 1;
            end
        end
        
        
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
        
        
        function trace_id = insert_spikes(key, S, channel, trace_id)
            
            for itrace = 1:size(S, 1)
                spike_key = struct(key);
                spike_key.channel = channel;
                spike_key.trace_id = trace_id;
                spike_key.spike_trace = single(S(itrace, :))';  % convert to single and make a column vector, see issue #115
                insert(preprocess.ExtractRawSpikeRate, spike_key);
                trace_id = trace_id + 1;
            end
            
        end
        
    end
    
    
end
