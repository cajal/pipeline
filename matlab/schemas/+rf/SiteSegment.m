%{
rf.SiteSegment (imported) # segmentation with matching trace IDs for each site
-> rf.ManualSegment
---
-> rf.Site
label_mask                   : longblob                      # label mask with label numbers shared across scans in the same site
segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}

classdef SiteSegment < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = (rf.Site*rf.Session & 'site>0') & (rf.Scan*rf.ManualSegment)
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            [masks,imgs,px,py,ux,uy,scanKeys] = fetchn(rf.ManualSegment*rf.Align*rf.ScanInfo*(rf.Scan & key), ...
                'mask','green_img','px_width','px_height','um_width','um_height');
            assert(all(px==px(1) & py==py(1) & ux==ux(1) & uy==uy(1)), 'all scans in site must be acquired at same resolution');
            px = ux(1)/px(1);  %  pixel pitch
            py = uy(1)/py(1);
            
            % align all masks using the green image
            n = length(masks);
            sz = size(masks{1});
            refIx = ceil(n/2);
            template = conj(fft2(imgs{refIx}));  % use the middle as reference
            minSeparation = 5; % microns
            
            for iScan=1:n
                map = zeros(sz,'uint16');
                [dx,dy] = ne7.ip.measureShift(fft2(imgs{iScan}).*template);
                props = regionprops(masks{iScan}, 'Centroid','PixelIdxList');
                cx = arrayfun(@(p) (p.Centroid(1) - dx).*px, props);
                cy = arrayfun(@(p) (p.Centroid(2) - dy).*py, props);
                if iScan == 1
                    centroids = [cx cy];
                    idx = 1:length(cx);
                else
                    for i=1:length(props)
                        % compute distances to existing centroids
                        [d2,j] = min(sum(bsxfun(@minus, centroids, [cx(i) cy(i)]).^2,2));
                        
                        if d2<minSeparation^2
                            % if distance is smaller than minSeparation, then count as existing
                            idx(i) = j;
                        else
                            % if new centroid, then add it to the list
                            centroids(end+1,:) = [cx(i) cy(i)]; %#ok<AGROW>
                            idx(i) = size(centroids,1);
                        end
                    end
                end
                
                % fill in mask
                for i=1:length(props)
                    map(props(i).PixelIdxList) = idx(i);
                end
                tuple = dj.struct.join(key,scanKeys(iScan));
                tuple.label_mask = map;
                self.insert(tuple)
            end
        end
    end
    
end