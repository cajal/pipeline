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
            if exists(anatomy.FieldCoordinates & key)
                background = cat(4,background,plot(anatomy.FieldCoordinates & key));
            end
            
            % create masks
            area_map = ne7.ui.paintMasks(abs(background));
            if isempty(area_map); disp 'No masks created!'; return; end
            
            % image
            masks = normalize(area_map);
            masks(:,:,2) = 0.2*(area_map>0);
            masks(:,:,3) = background(:,:,1,1);
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
        
        function plot(obj, back_idx)
            
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
                background = abs((background).^0.4);
                
            else
                % get maps
                background = getBackground(map.RetMap & (map.RetMapScan &  obj));
                
                % if FieldCoordinates exists add it to the background
                if exists(anatomy.FieldCoordinates & (mice.Mice & obj))
                    background = cat(4,background,plot(anatomy.FieldCoordinates & (mice.Mice & obj)));
                end
            end
            
            % merge masks with background
            im = hsv2rgb(cat(3,normalize(area_map),area_map>0,normalize(background(:,:,1,1))));
            if nargin<2 || isempty(back_idx) || back_idx > size(background,4)
                image((im));
            else
                imshowpair(im,background(:,:,:,back_idx),'blend')
            end
            
            % place area labels
            for iarea = 1:length(masks)
                s = regionprops(masks{iarea},'Centroid');
                text(s(1).Centroid(1),s(1).Centroid(2),areas{iarea})
            end
        end
    end
end