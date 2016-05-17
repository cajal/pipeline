%{
pre.ManualSegment (imported) # manual 2d cell segmentation$
-> pre.AlignMotion
---
mask                        : longblob                      # binary 4-connected mask image segmenting the aligned image
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}



classdef ManualSegment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.AlignMotion & pro(pre.AverageFrame)
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            images = fetchn(pre.AverageFrame & key & 'channel=1', 'frame', 'ORDER BY channel');
            assert(ismember(numel(images), [1 2]))
            bw = pre.ManualSegment.outlineCells(images);
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
    end
end
