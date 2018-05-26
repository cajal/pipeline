%{
# Area Masks for visualization
-> anatomy.Area
---
mask                     : mediumblob                       # area mask
%}

classdef Masks < dj.Lookup
    methods
        function plotMask(obj,color,n)
            [masks,areas]= fetchn(obj & 'brain_area!="MAP"' & 'brain_area!="unknown"','mask','brain_area');
            
            set(gcf,'name','Visual Areas')
            hold on
             set(gca,'ydir','reverse')
                set(gca,'xdir','reverse')
            axis image
            axis off
            
            for imask = 1:length(masks)
                stats = regionprops(masks{imask}) ;
                A = bwboundaries(masks{imask});
                if nargin>1
                    patch(A{1}(:,2),A{1}(:,1),color)
                else
                    plot(A{1}(:,2),A{1}(:,1),'k');
                end
                if nargin<3
                    text(stats.Centroid(1),stats.Centroid(2),...
                        sprintf('%s',areas{imask}),...
                        'horizontalalignment','center')
                else
                    text(stats.Centroid(1),stats.Centroid(2),...
                        sprintf('%s\n(%d)',areas{imask},n),...
                        'horizontalalignment','center')
                end
            end
            
        end
        
    end
end