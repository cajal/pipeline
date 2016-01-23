%{
pre.SelectedMask (computed) # selected masks from the tesselation
-> pre.SegmentationTile
-----
%}

classdef SelectedMask < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = pre.NMFSegment
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            [masks, tiles] = pre.SegmentationTile().fetch_scale_masks(key);
            [d1, d2] = fetch1(pre.ScanInfo() & key, 'px_height', 'px_width');
            M = reshape(masks, d1*d2, size(masks, 3));
            C = M'*M;
            
            selected = []; processed = [];
            for i = 1:size(masks, 3)
                if ~any(processed == i)
                    I = find(C(i,:) > 0.8);
                    R = [];
                    for j = I
                        [ii, jj] = find(masks(:,:,j));
                        row_center = mean([tiles(j).rstart, tiles(j).rend]);
                        col_center = mean([tiles(j).cstart, tiles(j).cend]);
                        r = sqrt( (mean(ii) - row_center)^2  + (mean(jj) - col_center)^2);
                        R = [R, r];
                    end
                    [~, sel] = min(R);
                    selected = [selected, I(sel)];
                    processed = [processed, I];
                end
            end
            self.insert(tiles(selected));
        end
    end
    
    methods(Static)
        %%------------------------------------------------------------
        function [masks, keys] = load_masks(key)
            [masks, keys] = pre.SegmentationTile().fetch_scale_masks(pre.SelectedMask() & key);
        end
    end
    
end