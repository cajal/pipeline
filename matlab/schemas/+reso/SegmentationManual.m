%{
# masks created manually
-> reso.Segmentation
---
mask                        : longblob                      # binary 4-connected mask image segmenting the aligned image
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}


classdef SegmentationManual < dj.Computed
    
    properties
        popRel  = reso.Segmentation & reso.SummaryImagesAverage
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            images = fetchn(reso.SummaryImagesAverage & key, 'average_image');
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