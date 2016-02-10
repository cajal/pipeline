%{
pre.ManualSegment (imported) # manual 2d cell segmentation$
-> pre.AlignMotion
---
mask                        : longblob                      # binary 4-connected mask image segmenting the aligned image
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}



classdef ManualSegment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.AlignMotion
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            template = fetch1(pre.AlignMotion & key, 'avg_frame');
            template = template - min(template(:));
            template = template / max(template(:));
            bw = pre.ManualSegment.outlineCells(template, false(size(template)));
            assert(~isempty(bw), 'user aborted segmentation')
            key.mask = bw;
            self.insert(key)
        end
    end
    
    
    methods(Static)
        function bw = outlineCells(imgG,bw)
            f = figure;
            imshow(imgG)
            set(gca, 'Position', [0.05 0.05 0.9 0.9]);
            if strcmp(computer,'GLNXA64')
                set(f,'Position',[160 160 1400 1000])
            end
            bw = ne7.ui.drawCells(bw);
            close(f)
        end
    end
end
