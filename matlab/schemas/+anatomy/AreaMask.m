%{
# Area mask for each scan
-> experiment.Scan
-> anatomy.Area
-> shared.Field
---
-> map.RetMap
mask                     : mediumblob            # mask of area
%}

classdef AreaMask < dj.Manual
    methods
        function createMasks(obj, key, varargin)
            
            params.exp = 1.5;
            params.sigma = 2;
            
            params = ne7.mat.getParams(params,varargin);
            
            % populate if retinotopy map doesn't exist
            ret_key = getRetKey(map.RetMap, key);
            
            % get maps
            background = getBackground(map.RetMap & ret_key, params);
            
            % if FieldCoordinates exists add it to the background
            ref_key = fetch(anatomy.RefMap & (map.RetMapScan & key));
            if exists(anatomy.FieldCoordinates & ref_key)
                background = cat(4,background,plot(anatomy.FieldCoordinates & ref_key));
            end
            
            % get masks already extracted
            if exists(obj & rmfield(key,'ret_idx'))
                [area_map, keys] = getContiguousMask(obj, rmfield(key,'ret_idx'));
            else
                area_map = zeros(size(background,1),size(background,2));
            end
            
            % create masks
            area_map = ne7.ui.paintMasks(abs(background),area_map);
            if isempty(area_map); disp 'No masks created!'; return; end
            
            % delete previous keys if existed
            if exists(obj & rmfield(key,'ret_idx'))
                del(anatomy.AreaMask & keys)
            end
            
            % image
            figure;
            masks = ne7.mat.normalize(area_map);
            masks(:,:,2) = 0.2*(area_map>0);
            masks(:,:,3) = background(:,:,1,1);
            ih = image(hsv2rgb(masks));
            axis image
            axis off
            shg
            
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
                
                % get base key
                tuple = ret_key;
                
                % get area name
                areas = fetchn(anatomy.Area,'brain_area');
                area_idx = listdlg('PromptString','Which area is this?',...
                    'SelectionMode','single','ListString',areas);
                tuple.brain_area = areas{area_idx};
                if ~isfield(tuple,'field')
                    tuple.field = 1;
                end
                tuple.mask = area_map == iarea;
                th.delete;
                insert(obj,tuple)
                
                % set correct area label
                text(s.Centroid(1),s.Centroid(2),tuple.brain_area)
            end
        end
        
        function extractMasks(obj, keyI)
            
            % fetch all area masks
            map_keys = fetch(anatomy.AreaMask & (anatomy.RefMap & (proj(anatomy.RefMap) & (anatomy.FieldCoordinates & keyI))));
            
            % loop through all masks
            for map_key = map_keys'
                [mask, area, ret_idx] = fetch1(anatomy.AreaMask & map_key, 'mask', 'brain_area', 'ret_idx');
                
                % loop through all fields
                for field_key = fetch(anatomy.FieldCoordinates & keyI)'
                    
                    % find corresponding mask area
                    fmask = filterMask(anatomy.FieldCoordinates & field_key, mask);
                    
                    % insert if overlap exists
                    if ~all(~fmask(:))
                        tuple = rmfield(field_key,'ref_idx');
                        tuple.brain_area = area;
                        tuple.mask = fmask;
                        tuple.ret_idx = ret_idx;
                        insert(obj,tuple)
                    end
                end
            end
        end
        
        function plot(obj, varargin)
            
            params.back_idx = [];
            params.exp = 0.4;
            
            params = ne7.mat.getParams(params,varargin);
            
            % get mask info
            [masks, areas] = fetchn(obj,'mask','brain_area');
            
            % identify mask areas
            area_map = zeros(size(masks{1}));
            for imasks = 1:length(masks)
                area_map(masks{imasks}) = imasks;
            end
            
            if exists(fuse.ScanDone & obj)
                % fetch 2p avg image
                if strcmp(fetch1(experiment.Session & obj, 'rig'),'2P4')
                    background = fetch1(meso.SummaryImagesAverage & (fuse.ScanSet & obj),...
                        'average_image');
                else
                    background = fetch1(reso.SummaryImagesAverage & (fuse.ScanSet & obj),...
                        'average_image');
                end
            else
                % get maps
                background = getBackground(map.RetMap & (map.RetMapScan &  obj));
                
                % if FieldCoordinates exists add it to the background
                if exists(anatomy.FieldCoordinates & (mice.Mice & obj))
                    background = cat(4,background,plot(anatomy.FieldCoordinates &...
                        (anatomy.RefMap & obj)));
                end
            end
            
            % adjust background contrast
            background = ne7.mat.normalize(abs(ne7.mat.normalize(background).^ params.exp));
            
            % merge masks with background
            im = hsv2rgb(cat(3,ne7.mat.normalize(area_map),area_map>0,background(:,:,1,1)));
            if nargin<2 || isempty(params.back_idx) || params.back_idx > size(background,4)
                image((im));
            else
                imshowpair(im,background(:,:,:,params.back_idx),'blend')
            end
            
            % place area labels
            for iarea = 1:length(masks)
                s = regionprops(masks{iarea},'Centroid');
                text(s(1).Centroid(1),s(1).Centroid(2),areas{iarea})
            end
        end
        
        function [area_map, keys] = getContiguousMask(obj, key)
            
            % fetch masks & keys
            [masks, keys] = fetchn(obj & key,'mask');
            
            % get information from the scans depending on the setup
            if strcmp(fetch1(experiment.Session & key,'rig'),'2P4') || length(masks)<2
                [x_pos, y_pos, fieldWidths, fieldHeights, fieldWidthsInMicrons, masks, areas] = ...
                    fetchn(obj * meso.ScanInfoField * meso.SummaryImagesAverage & key,...
                    'x','y','px_width','px_height','um_width','mask','brain_area');
                
                % calculate initial scale
                pxpitch = mean(fieldWidths.\fieldWidthsInMicrons);
                
                % construct a big field of view
                x_pos = (x_pos - min(x_pos))/pxpitch;
                y_pos = (y_pos - min(y_pos))/pxpitch;
                area_map = zeros(ceil(max(y_pos+fieldHeights)),ceil(max(x_pos+fieldWidths)));
                for islice =length(masks):-1:1
                    mask = double(masks{islice})*find(strcmp(areas{islice},unique(areas)));
                    y_idx = ceil(y_pos(islice)+1):ceil(y_pos(islice))+size(mask,1);
                    x_idx = ceil(x_pos(islice)+1):ceil(x_pos(islice))+size(mask,2);
                    back = area_map(y_idx, x_idx);
                    area_map(y_idx, x_idx) = max(cat(3,mask,back),[],3);
                end
                
            else
                area_map = zeros(size(masks{1}));
                for imasks = 1:length(masks)
                    area_map(masks{imasks}) = imasks;
                end
            end
        end
    end
end