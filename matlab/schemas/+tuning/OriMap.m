%{
# pixelwise responses to full-field directional stimuli
-> tuning.OriDesignMatrix
-> preprocess.PrepareGalvo
-> preprocess.Slice
---
regr_coef_maps              : longblob                      # regression coefficients, width x height x nConds
r2_map                      : longblob                      # pixelwise r-squared after gaussinization
dof_map                     : longblob                      # degrees of in original signal, width x height
%}


classdef OriMap < dj.Imported

	properties(Constant)
		keySource = tuning.OriDesignMatrix*preprocess.PrepareGalvo*preprocess.Slice & 'slice>=1 and slice<=nslices'
	end

	methods(Access=protected)

		function makeTuples(self, key)
            tic
            disp 'loading design...'
            designMatrix = fetch1(tuning.OriDesignMatrix & key, 'design_matrix');
                       
            disp 'loading movie...'
            fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
            if exists(preprocess.PrepareGalvoMotion & key)
                fixMotion = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
            else
                % no motion correction
                fixMotion = @(img, i) img;
            end
            [height, width, nslices] = fetch1(preprocess.PrepareGalvo & key, 'px_height', 'px_width', 'nslices');
            designMatrix = designMatrix(key.slice:nslices:end,:);
            designMatrix = bsxfun(@minus, designMatrix, mean(designMatrix));
            reader = preprocess.getGalvoReader(key);
            nframes = reader.nframes;
            assert(size(designMatrix,1)==nframes, 'movie reads incorrectly')
            frames = any(designMatrix, 2);
            frames = find(frames, 1, 'first'):find(frames, 1, 'last');
            designMatrix = designMatrix(frames,:);
            X = zeros(length(frames), height*width, 'single');            
            for iframe=1:length(frames)
                if ismember(iframe,[1 10 100 500 1000 5000 nframes]) || mod(iframe,10000)==0
                    fprintf('Frame %5d/%d  %4.1fs\n', iframe, nframes, toc);
                end
                frame = fixMotion(fixRaster(double(reader(:,:,1,key.slice,frames(iframe)))), frames(iframe));
                X(iframe,:) = frame(:);
            end
            
            % remove periods where the design matrix has nans
            ix = any(isnan(X),2);
            X(ix,:) = [];
            designMatrix(ix,:) = [];
            
            % normalize traces
            M = mean(X);
            X = bsxfun(@minus, X, M);
            X = bsxfun(@rdivide, X, M);
            
            disp inverting..
            [B, R2, ~, DoF] = ne7.stats.regress(X, designMatrix, 0);

            disp inserting..
            tuple = key;
            tuple.regr_coef_maps = reshape(B', width, height,[]);
            tuple.r2_map = reshape(R2, width, height);
            tuple.dof_map = reshape(DoF, width, height);
            self.insert(tuple)
		end
    end
    
end
