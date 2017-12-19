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
            %             insert( obj, key );
        end
    end
    
    methods
        function createMasks(self,key,varargin)
            params.exp = 1.5;
            params.sigma = 2;
            
            params = ne7.mat.getParams(params,varargin);
            
            % get maps
            Hor = [];Ver = [];
            opt_key = fetch(map.OptImageBar & (map.RetMapScan & key) & 'axis="horizontal"');
            [Hor(:,:,1),Hor(:,:,2),Hor(:,:,3)] = plot(map.OptImageBar & (map.RetMapScan & key) & 'axis="horizontal"','exp',params.exp,'sigma',params.sigma);
            [Ver(:,:,1),Ver(:,:,2),Ver(:,:,3)] = plot(map.OptImageBar & (map.RetMapScan & key) & 'axis="vertical"','exp',params.exp,'sigma',params.sigma);
            background = cat(4,hsv2rgb(Hor),hsv2rgb(Ver));
            if exists(map.SignMap & key)
                sign_map = fetch1(map.SignMap & key,'sign_map');
                background = cat(4,background,sign_map);
            end
            
            % create masks
            area_map = ne7.ui.paintMasks(abs(background));
            
            if ~isempty(area_map)
                % image
                masks = normalize(area_map);
                masks(:,:,2) = 0.2*(area_map>0);
                masks(:,:,3) = Hor(:,:,3);
                ih = image(hsv2rgb(masks));
                axis image
                axis off

                % loop through all areas get area name and insert
                areas = unique(area_map(:));
                for iarea = areas(2:end)'

                    % fix saturation for selected area
                    colors =  0.2*(area_map>0);
                    colors(area_map==iarea) = 1;
                    masks(:,:,2) = colors;
                    ih.CData = hsv2rgb(masks);
                    shg
                    s = regionprops(area_map==iarea,'area','Centroid');
                    th = text(s.Centroid(1),s.Centroid(2),'?');

                    tuple = rmfield(opt_key,'axis');

                    % get area name
                    areas = fetchn(anatomy.Area,'brain_area');
                    area_idx = listdlg('PromptString','Which area is this?',...
                    'SelectionMode','single','ListString',areas);
                    tuple.brain_area = areas{area_idx};
                    if ~isfield(tuple,'field')
                        tuple.field = 1;
                    end
                    tuple.mask = area_map==iarea;
                    th.delete;
                    insert(self,tuple)

                    % set correct area label
                    text(s.Centroid(1),s.Centroid(2),tuple.brain_area)

                end
            end
            
        end
        
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