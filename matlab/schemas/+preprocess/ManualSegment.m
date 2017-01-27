%{
preprocess.ManualSegment (imported) # manual 2d cell segmentation$
-> preprocess.PrepareGalvoMotion
---
mask                        : longblob                      # binary 4-connected mask image segmenting the aligned image
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}



classdef ManualSegment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = preprocess.PrepareGalvoMotion & pro(preprocess.PrepareGalvoAverageFrame)
    end
    
    methods
        % Overloaded delete function:
        % Deletes manual segmentations for other slices of this scan
        % Deletes dependents of all manual segmentations for all slices of this scan
        function del(self)
            
            % Check for any dependent data
            if count(preprocess.ExtractRaw & self & 'extract_method=1')
                
                % There is dependent data. Prompt for delete
                del(preprocess.ExtractRaw & self & 'extract_method=1');
                
                % only if dependent data is gone, prompt to delete the segmentations
                if ~count(preprocess.ExtractRaw & self & 'extract_method=1')
                    del@dj.Relvar(self);
                end
            else
                % No dependent data - can delete segmentations
                del@dj.Relvar(self);
            end
            
            
        end
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            images = fetchn(preprocess.PrepareGalvoAverageFrame & key, 'frame', 'ORDER BY channel');
            assert(ismember(numel(images), [1 2]))
            if verLessThan('matlab', '9.1')
                warning('You are running an older version of Matlab, switchin to the old segmenation code!')
                bw = preprocess.ManualSegment.outlineCells(images);
            else
                bw = preprocess.ManualSegment.paintCells(images);
            end
            assert(~isempty(bw), 'user aborted segmentation')
            key.mask = bw;
            self.insert(key)
        end
    end
    
    methods(Static)
        function bw = outlineCells(images, bw)
            if ~exist('bw','var')
                bw = false(size(images{1}));
            end
            f = figure;
            if length(images)==2
                imshowpair(sqrt(images{1}), sqrt(images{2}))
            else
                template = sqrt(images{1});
                template = template - min(template(:));
                template = template / max(template(:));
                imshow(template)
            end
            set(gca, 'Position', [0.05 0.05 0.9 0.9]);
            if strcmp(computer,'GLNXA64')
                set(f,'Position',[160 160 1400 1000])
            end
            bw = ne7.ui.drawCells(bw);
            close(f)
        end
        
        function bw = paintCells(images)
            bw = ne7.ui.paintMasks(images{1});
        end
    end
end
