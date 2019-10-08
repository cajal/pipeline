%{
# Area Masks for visualization
-> anatomy.Area
---
mask                     : mediumblob                       # area mask
%}

classdef Masks < dj.Lookup
    methods
        function plotMask(obj,color,n,varargin)
            params.fontsize = 10;
            params.outline = [];
            
            params = ne7.mat.getParams(params,varargin);
            
            [masks,areas]= fetchn(obj & 'brain_area!="MAP"' & 'brain_area!="unknown"' & 'brain_area<>"A"','mask','brain_area');
            
            set(gcf,'name','Visual Areas')
            hold on
             set(gca,'ydir','reverse')
                set(gca,'xdir','reverse')
            axis image
            axis off
            
            for imask = 1:length(masks)
                stats = regionprops(masks{imask}) ;
                A = bwboundaries(masks{imask});
                if nargin>1 && ~isempty(color)
                    patch(A{1}(:,2),A{1}(:,1),color)
                end
                
                if ~isempty(params.outline)
                    plot(A{1}(:,2),A{1}(:,1),'color',params.outline);
                end
                
                if nargin<3 || isempty(n)
                    text(stats.Centroid(1),stats.Centroid(2),...
                        sprintf('%s',areas{imask}),...
                        'horizontalalignment','center','fontsize',params.fontsize)
                else
                    text(stats.Centroid(1),stats.Centroid(2),...
                        sprintf('%s\n(%d)',areas{imask},n),...
                        'horizontalalignment','center','fontsize',params.fontsize)
                end
            end
            
        end
        
    end
end