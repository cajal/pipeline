%{
rf.OriMap (imported) # orientation tuning
-> rf.OriDesign
-> rf.VolumeSlice
-----
regr_coef_maps              : longblob                      # regression coefficients, width x height x nConds
r2_map                      : longblob                      # pixelwise r-squared after gaussinization
dof_map                     : longblob                      # degrees of in original signal, width x h
%}

classdef OriMap < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = rf.OriDesign * rf.VolumeSlice
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            disp 'loading movie...'
            reader = rf.getReader(key);
            [fillFraction, rasterPhase] = fetch1(rf.Align & key, 'fill_fraction', 'raster_phase');
            iSlice = key.slice_num;
            reader.reset
            blockSize = 1000;
            xymotion = fetch1(rf.Align & key, 'motion_xy');
            [width,height,nFrames] = fetch1(rf.Align*rf.ScanInfo & key, ...
                'px_width','px_height','nframes');
            X = nan(nFrames,width*height,'single');
            lastPos = 0;
            while ~reader.done
                block = getfield(reader.read(1, iSlice, blockSize),'channel1'); %#ok<GFLD>
                sz = size(block);
                xy = xymotion(:,:,1:sz(4));
                xymotion(:,:,1:size(block,4)) = [];
                block = ne7.ip.correctRaster(block,rasterPhase,fillFraction);
                block = ne7.ip.correctMotion(block, xy);
                X(lastPos+(1:sz(4)),:) = reshape(block,[],sz(4))';
                lastPos = lastPos + sz(4);
                fprintf('frame %4d\n',lastPos);
            end
            clear block
            
            disp 'loading design matrix...'
            assert(~any(any(isnan(X))))
            fps = fetch1(rf.ScanInfo & key, 'fps');
            G = fetch1(rf.OriDesign & key, 'design_matrix');
            X = X(1:min(size(G,1),size(X,1)),:);
            G = G(1:size(X,1),:);
            
            B = zeros(size(G,2),size(X,2),'single');
            R2 = zeros(1,size(X,2),'single');
            DoF = zeros(1,size(X,2),'single');
            chunkSize = 128;
            
            disp 'computing responses (in chunks)...'
            highpass = 0.04;
            k = hamming(round(fps/highpass)*2+1);
            k = k/sum(k);
            for i=1:chunkSize:size(X,2)-1
                ix = i:min(size(X,2),i+chunkSize-1);
                X_ = double(X(:,ix));                
                X_ = bsxfun(@rdivide, X_, mean(X_))-1;  %use dF/F
                X_ = X_ - ne7.dsp.convmirr(X_,k);
                fprintf .
                [B(:,ix),R2(ix),~,DoF(ix)] = ne7.stats.regress(X_, G, 0);
            end
            fprintf \n
            
            disp 'inserting...'
            tuple = key;
            tuple.regr_coef_maps = reshape(B', width, height,[]);
            tuple.r2_map = reshape(R2, width, height);
            tuple.dof_map = reshape(DoF, width, height);
            self.insert(tuple)
        end
    end
end