%{
map.Masks (imported) #
-> map.Area
---
mask                     : mediumblob                       # area mask
%}

classdef Masks < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = (map.Area)
    end
    
    methods (Access=protected)
        function makeTuples(obj,key)
            
            path = 'C:\Users\Manolis\Desktop\Areas';
            areas = fetchn(map.Area,'area');
            
            for iarea = 1:length(areas)
                
                file = fullfile(path,[areas{iarea},'.png']);
                
                data = importdata(file);
                
                tuple.area = areas{iarea};
                tuple.mask = im2bw(data.alpha);
                
                % insert Mask
                insert( obj, tuple );
                
            end
            
            
        end
    end
    
    methods
        function self = Scan(varargin)
            self.restrict(varargin{:})
        end
        
        function plotMask(obj,color,n)
            [masks,areas]= fetchn(obj & 'area!="MAP"','mask','area');
            
            %             figure
            set(gcf,'name','Visual Areas')
            hold on
            set(gca,'ydir','reverse')
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