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
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            %% !!! compute missing fields for key here
            if ~count(preprocess.PrepareGalvo & key)
                
                
                [d2, d1, um_width, um_height, nslices] = fetch1(preprocess.PrepareGalvo & key, ...
                    'px_width', 'px_height', 'um_width', 'um_height','nslices');
                
                if strcmp(fetch1(experiment.Scan & key, 'aim'),'unset') | ...
                        strcmp(fetch1(experiment.Scan & key, 'aim'), 'functional: somas')
                    tau = 4;
                    p = 2;
                    neuron_density = 800; % neuron per mm^2 slice
                    max_iter = 2;
                    downsample_to = 4;
                    options = CNMFSetParms(...
                        'd1',d1,'d2',d2,...                         % dimensions of datasets
                        'search_method','ellipse','dist',3,...      % search locations when updating spatial components
                        'deconv_method','constrained_foopsi',...    % activity deconvolution method
                        'temporal_iter',2,...                       % number of block-coordinate descent steps
                        'fudge_factor',0.98,...                     % bias correction for AR coefficients
                        'merge_thr',.8,...                          % merging threshold
                        'gSig', tau, ...
                        'se',  strel('disk',3,0),...
                        'init_method', 'greedy_corr', ...
                        'tsub', floor(fetch1(preprocess.PrepareGalvo & key, 'fps')/downsample_to) ...
                        );
                    
                elseif strcmp(fetch1(experiment.Scan & key, 'aim'),'functional: axons')
                    error('Segmentation for axons not implemented yet')
                end
                max_neurons = round((um_width * um_height)/1000^2 * neuron_density);
                fprintf('Using max %i neurons\n', max_neurons);
                
               
                self.insert(key);
                trace_id = 1;
                channel = 1;  % TODO: change to more flexible choice
                
                for islice = 1:nslices
                    Y = squeeze(self.load_galvo_scan(key, islice, channel));
                    [A,S] = self.run_nmf(Y, max_neurons, tau, p, max_iter,  options);
                    
                    Yr = reshape(Y,d1*d2,size(Y,3));
                    traces = A'*Yr;
                    
                    seg_key = struct(key);
                    seg_key.slice = islice;
                    insert(preprocess.ExtractRawGalvoSegmentation, seg_key);
                    
                    for itrace = 1:size(traces, 1)
                        trace_key = struct(key);
                        trace_key.channel = channel;
                        trace_key.trace_id = trace_id;
                        trace_key.raw_trace = traces(itrace, :);
                        insert(preprocess.ExtractRawTrace, trace_key);
                      
                        
                        spike_key = struct(key);
                        spike_key.channel = channel;
                        spike_key.trace_id = trace_id;
                        spike_key.spike_trace = S(:, itrace);
                        insert(preprocess.ExtractRawSpikeRate, spike_key);
                        
                        
                        mask_key = struct(seg_key);
                        mask_key.channel = channel;
                        mask_key.trace_id =  trace_id;
                        I = find( A(:,itrace));
                        mask_key.mask_pixels = I;
                        mask_key.mask_weights = A(I, itrace);
                        insert(preprocess.ExtractRawGalvoROI, mask_key);
                        
                        trace_id = trace_id + 1;
                    end
                    
                    clear Yr;
                end

                %% extract AOD
            elseif ~count(preprocess.PrepareAod & key)
                error('AOD trace extraction not implemented.')
            else
                error('Cannot match scan to neither Galvo nor AOD.')
            end
            % 			self.insert(key)
        end
    end
    
    methods(Static)
        
        function [A,S] = run_nmf(Y, K, tau, p, max_iter, options)
            %
            % Runs the nonnegative matrix factorization algorithm on Y with configuration cfg.
            %
            
            [d1,d2, T] = size(Y);
            d = d1*d2;
            
            [P,Y] = preprocess_data(Y,p);
            
            % fast initialization of spatial components using greedyROI and HALS
            [A,C,b,f,center] = initialize_components(Y,K,tau,options,P);  % initialize
            
            % update spatial and temporal components
            Yr = reshape(Y,d,T);
            clear Y;
            
            % update spatial components
            for iter = 1:max_iter
                [A,b,C] = update_spatial_components(Yr,C,f,A,P,options);
                [C,f,P,S] = update_temporal_components(Yr,A,b,C,f,P,options);
                % merge found components
                [A,C,~,~,P,S] = merge_components(Yr,A,b,C,f,P,S,options);
            end
            [A,C,S,P] = order_ROIs(A,C,S,P);    % order components
        end
        
        
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
        
        
    end
    
    
end