%{
rf.ManualSegment (imported) # manual 2d cell segmentation$
-> rf.Align
-> rf.VolumeSlice
---
mask                        : longblob                      # binary 4-connected mask image segmenting the aligned image
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}



classdef ManualSegment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = rf.Align
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            g = fetch1(rf.Align & key, 'green_img');
            sz = size(g);
            
            fluo = fetch1(rf.Session & key,'fluorophore');
            if strcmp(fluo,'TN-XXL')
                r = fetch1(rf.Align & key, 'red_img');
            else
                r = nan(sz);
            end
            
            bw = false(sz(1:2));
            for iSlice = 1:size(g,3)
                key.slice_num = iSlice;
                bw = rf.ManualSegment.outlineCells(g(:,:,iSlice),r(:,:,iSlice),bw);
                assert(~isempty(bw), 'user aborted segmentation')
                key.mask = bw;
                self.insert(key)
            end
        end
    end
    
    
    methods(Static)
        function bw = outlineCells(imgG,imgR,bw)
            f = figure;
            
            if any(isnan(imgR(:)))
                imshow(imgG)
            else
                figure(f)
                imgG=imgG-min(imgG(:)); imgR=imgR-min(imgR(:));
                imgG = imgG./max(imgG(:)); imgR = imgR./max(imgR(:));
                imshow(imgG)
            end
            
            
            set(gca, 'Position', [0.05 0.05 0.9 0.9]);
            pos = get(f, 'Position');
            if strcmp(computer,'GLNXA64')
                set(f,'Position',[160 160 1400 1000])
            else
                %set(f, 'Position', [pos(1:2)/4 pos(3:4)*4])
            end
            bw = ne7.ui.drawCells(bw);
            close(f)
        end
    end
end
