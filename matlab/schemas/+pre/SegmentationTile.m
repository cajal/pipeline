%{
pre.SegmentationTile (computed) # my newest table
-> pre.Tesselation
idx                    : int  # index of the mask and its spike trace
-----
mask                 : longblob # weighted inferred neuron mask
spiketrace           : longblob # inferred spike trace
p                    : int      # order of AR process
gn                   : longblob # parameters of AR process
%}

classdef SegmentationTile < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.Tesselation()
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            cfg = fetch(pre.Settings & key,'*');
            [d1, d2, nslices] = self.get_resolution(key);
            assert(nslices==1, 'This schema only supports one slice.')
            
            [um_width, um_height] = fetch1(pre.ScanInfo & key, 'um_width', 'um_height');
            px_r = (key.rend - key.rstart + 1);
            px_c = (key.cend - key.cstart + 1);
            um_w = px_c/d2 * um_width;
            um_h = px_r/d1 * um_height;
            cfg.max_neurons = round((um_w * um_h)/1000^2 * cfg.density);
            
            fprintf('Using max %i neurons\n',cfg.max_neurons);
            
            % make sure a 6 um diameter cell body is at least covered by 2x2 pixels
            % use a power of two
            scale = min(1, 2^nextpow2(2 / 6 * max([um_width/d2, um_height/d1])));
            fprintf('Scaling image by a factor of %.2f\n',scale);
            
            Y = squeeze(self.load_scan(key, scale, key.rstart:key.rend, key.cstart:key.cend));
        
            params = self.run_nmf(Y, cfg);
            
            
            A = full(params.A);
            A = reshape(A, scale*px_r, scale*px_c, size(A,2)); % reshape into masks
            A = imresize(A, 1/scale, 'lanczos2'); % upscale
            A = bsxfun(@rdivide, A, sqrt(sum(sum(A.^2, 1),2))); % normalize to norm 1
            
            for idx = 1:size(A,3)
                key.idx = idx;
                key.p = params.p(idx);
                key.gn = params.gn(idx);
                
                % insert mask
                key.mask = A(:,:,idx);
               
                
                % insert spiketrace
                key.spiketrace = params.S(idx,:);
                
                self.insert(key);
            end
        end
    end
    
    
    methods(Static)
        %%------------------------------------------------------------
        function [masks, keys] = fetch_scale_masks(key)
            [d1, d2] = fetch1(pre.ScanInfo() & key, 'px_height', 'px_width');
            keys = fetch(pre.SegmentationTile & key,'mask');
            masks = zeros(d1, d2, length(keys));
            for i = 1:length(keys)
                masks(keys(i).rstart:keys(i).rend,keys(i).cstart:keys(i).cend,i)  = keys(i).mask;
            end
            keys = rmfield(keys, 'mask');
        end
        
        %%------------------------------------------------------------
        function [d1, d2, nslices] = get_resolution(key)
            [d1,d2, nslices] = fetch1(pre.ScanInfo & key, 'px_height', 'px_width','nslices');
        end
        %%------------------------------------------------------------
        
        function params = run_nmf(Y, cfg)
            [d1,d2, T] = size(Y);
            d = d1*d2;
            
            K = cfg.max_neurons;    % number of components to be found
            tau = cfg.tau;          % std of gaussian kernel (size of neuron)
            p = cfg.p;              % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
            
            
            options = CNMFSetParms(...
                'd1',d1,'d2',d2,...
                'search_method',cfg.search_method ,'dist',cfg.dist,...
                'deconv_method',cfg.deconv_method,...
                'temporal_iter',cfg.temporal_iter,...
                'fudge_factor',cfg.fudge_factor,...
                'merge_thr',cfg.merge_thr,...
                'se', strel('disk',cfg.se,0) ...
                );
            %% Data pre-processing
            
            [P,Y] = preprocess_data(Y,p);
            
            %% fast initialization of spatial components using greedyROI and HALS
            [A,C,~,f] = initialize_components(Y,K,tau,options);  % initialize
            
            %% update spatial and temporal components
            Yr = reshape(Y,d,T);
            clear Y;
            
            for i = 1:cfg.max_iter
                % update spatial components
                [A,b] = update_spatial_components(Yr,C,f,A,P,options);
                % update temporal components
                [C,f,Y_res,P,S] = update_temporal_components(Yr,A,b,C,f,P,options);
                
                % merge found components
                [A,C,K,~,P,S] = merge_components(Y_res,A,b,C,f,P,S,options);
            end
            [A,C,S,P] = order_ROIs(A,C,S,P);    % order components
            [C,~,S] = extract_DF_F(Yr,[A,b],[C;f],S,K+1); % extract dF/F values
            
            p = cellfun(@length, P.gn);
            gn = P.gn;
            
            params = struct('A', A, 'C', C, 'S', S);
            params.p = p;
            params.gn = gn;
            
        end
        
        
        %%------------------------------------------------------------
        function scan = load_scan(key, scale, rows, cols, maxT, blockSize)
            reader = pre.getReader(key, '/tmp');
            assert(reader.nslices == 1, 'schema only supports one slice at the moment');
            
            if nargin < 5
                maxT = reader.nframes;
            end
            
            if nargin < 6
                blockSize = min(maxT, 10000);
            end
            
            % get TIFF reader for key into rf.Scan
            
            % For compatibility with Piezo Z-scanning, get Z-slice keys for each scan
            % For the case without the piezo, sliceKeys is the same as key, but adds primary key attribute VolumeSlice = 1
            [r,c] = fetch1(pre.ScanInfo & key, 'px_height', 'px_width');
            
            if nargin < 3
                rows = 1:r;
                cols = 1:c;
            end
            
            % if no rows and columns are specified, take the full image
            if nargin <  2
                scale = 1;
            end
            
            downsample = 1/scale;
            scan = zeros(length(rows)/downsample,length(cols)/downsample, 1, maxT);

            pointer = 1;
            while pointer < maxT
                step =  min(blockSize, maxT-pointer+1);
                frames = pointer:pointer+step-1;
                fprintf('Reading block (%i:%i, %i:%i, %i:%i) of maximally %i (video has %i frames)\n', ...
                    rows(1), rows(end), cols(1), cols(end), pointer, pointer + step - 1, maxT, reader.nframes);
                tmp_scan = pre.load_corrected_block(key, reader, frames);
                
                if scale == 1
                    scan(:, :, 1, frames) = tmp_scan(rows, cols,1,:);
                else
                    scan(:, :, 1, frames) = imresize(tmp_scan(rows, cols ,1,:), scale, 'lanczos2');
                end
                pointer = pointer + step;
            end
            
        end
    end
end