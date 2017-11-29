%{
# Area mask for each scan
-> experiment.Scan
-> anatomy.Area
-> shared.Field
---
mask                     : mediumblob            # mask of area
%}

classdef AreaMask < dj.Imported
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
%            insert( obj, key );
        end
    end
    
    methods
        
        function plot(obj)
            
            [masks,areas] = fetchn(obj,'mask','area');
            vessels = fetchn(map.OptImageBar & obj,'vessels');
            
            area_map = zeros(size(masks{1}));
            for imasks = 1:length(masks)
                area_map(masks{imasks}) = imasks;
            end
            
            im = cat(3,normalize(area_map),area_map>0,normalize(vessels{1}));
            image(hsv2rgb(im));
            
            for iarea = 1:length(masks)
                s = regionprops(masks{iarea},'Centroid');
                text(s(1).Centroid(1),s(1).Centroid(2),areas{iarea})
            end
        end
        
    end
    
    methods(Static)
        function extractMask(keyI,keyV)
            
            if nargin<2
                keyV = fetch(map.OptImageBar & anatomy.AreaMask & sprintf('animal_id=%d',keyI.animal_id));
            end
            
            % Insert overlaping masks
            map_keys = fetch(anatomy.AreaMask & keyV);
            for map_key = map_keys'
                [mask, area] = fetch1(anatomy.AreaMask & map_key,'mask','area');
                for tuple = fetch(anatomy.FieldCoordinates & keyI)'
                    fmask = filterMask(anatomy.FieldCoordinates & tuple,mask);
                    if ~all(~fmask(:))
                        tuple.area = area;
                        tuple.mask = fmask;
                        makeTuples(anatomy.AreaMask,tuple)
                    end
                end
            end
        end
        
    end
end