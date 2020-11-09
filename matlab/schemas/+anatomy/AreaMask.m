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
        function createMasks(self, key, varargin)
            
            params.exp = 1.5;
            params.sigma = 2;
            params.amp = 1;
            params.downsample = 1;
            
            params = ne7.mat.getParams(params,varargin);
            
            % populate if retinotopy map doesn't exist
            ret_key = getRetKey(map.RetMap, key);
%             key = rmfield(ret_key,'ret_idx');
            
            % get maps
            background = getBackground(map.RetMap & ret_key, params);
            
            % if FieldCoordinates exists add it to the background
            ref_key = fetch(anatomy.RefMap & (map.RetMapScan & key));
            if exists(anatomy.FieldCoordinates & ref_key)
                background = cat(4,background,plot(anatomy.FieldCoordinates & ref_key));
            end
            
            % get masks already extracted
            if strcmp(fetch1(experiment.Scan & key,'aim'),'widefield') || strcmp(fetch1(experiment.Scan & key,'aim'),'intrinsic')
                contiguous = 1;
            else
                contiguous = 0;
            end
            
            if exists(self & key)
                [area_map, keys] = getContiguousMask(self, key,contiguous);
            else
                area_map = zeros(size(background,1),size(background,2));
            end
            
            % donwscale
            if params.downsample ~= 1
                background = imresize(background,params.downsample);
                area_map = (uint8(imresize(area_map,params.downsample,'nearest')));
            end
            
            % create masks
            area_map = ne7.ui.paintMasks(abs(background),area_map);
            if isempty(area_map); disp 'No masks created!'; return; end
            
            % upscale
            if params.downsample ~= 1
                area_map = single(uint8(imresize(area_map,1/params.downsample,'nearest')));
            end
            
%             % delete previous keys if existed
%             if exists(self & key)
%                 del(anatomy.AreaMask & keys)
%             end
            
            % image
            figure;
            masks = ne7.mat.normalize(area_map);
            masks(:,:,2) = 0.2*(area_map>0);
            %             masks(:,:,3) = background(:,:,1,1);
            masks(:,:,3) = ones(size(masks(:,:,2)));
            ih = image(hsv2rgb(masks));
            axis image
            axis off
            shg
            
            % loop through all areas get area name and insert
            areas = unique(area_map(:));
            areas = areas(areas>0);
            brain_areas = [];
            for iarea = areas'
                % fix saturation for selected area
                colors =  0.2*(area_map>0);
                colors(area_map==iarea) = 1;
                masks(:,:,2) = colors;
                ih.CData = hsv2rgb(masks);
                s = regionprops(area_map==iarea,'area','Centroid');
                th = text(s(1).Centroid(1),s(1).Centroid(2),'?');
                shg
                
                % ask for area name
                areas = fetchn(anatomy.Area,'brain_area');
                area_idx = listdlg('PromptString','Which area is this?',...
                    'SelectionMode','single','ListString',areas);
                brain_areas{iarea} = areas{area_idx};
                th.delete;
                
                % set correct area label
                text(s(1).Centroid(1),s(1).Centroid(2),brain_areas{iarea})
            end
            
            % get base key
            tuple = ret_key;
            
            if ~contiguous
                % get field specific area map
                [field_area_maps, fields] = splitContiguousMask(self, tuple, area_map);
            else
                field_area_maps{1} = area_map;
                fields(1) = 1;
            end
            
            % loop through all fields
            for ifield = 1:length(fields)
                tuple.field = fields(ifield);
                
                % loop through all areas get area name and insert
                areas = unique(field_area_maps{ifield}(:));
                areas = areas(areas>0);
                for iarea = areas'
                    
                    % get area name
                    tuple.brain_area = brain_areas{iarea};
                    tuple.mask = field_area_maps{ifield} == iarea;
                    insert(self,tuple,'REPLACE')
                end
            end
        end
        
        function extractMasks(obj, keyI, contiguous)
            
            % fetch all area masks
            area_masks = [];areas = [];
            keyI.ret_idx = fetch1(map.RetMap & keyI,'ret_idx');
            map_keys = fetch(anatomy.AreaMask & (anatomy.RefMap & (proj(anatomy.RefMap) & (anatomy.FieldCoordinates & keyI))));
            if nargin >2 && contiguous
                % if contiguous
                [area_map, keys] = getContiguousMask(obj,map_keys,0);
                areas = unique({keys(:).brain_area}');
                un_areas = unique(area_map(:));
                for iarea = 2:length(un_areas)
                    area_masks{end+1} = area_map==un_areas(iarea);
                end
            else
                for map_key = map_keys'
                    [mask, area, ret_idx] = fetch1(anatomy.AreaMask & map_key, 'mask', 'brain_area', 'ret_idx');
                    area_masks{end+1} = mask;
                    areas{end+1} = area;
                end
                
            end
            
            % loop through all masks
            for imask = 1:length(area_masks)
                % loop through all fields
                for field_key = fetch(anatomy.FieldCoordinates & keyI)'
                    
                    % find corresponding mask area
                    fmask = ne7.mat.normalize(filterMask(anatomy.FieldCoordinates & field_key, area_masks{imask}))>0;
                    
                    % insert if overlap exists
                    if ~all(~fmask(:))
                        tuple = rmfield(field_key,'ref_idx');
                        tuple.brain_area =   areas{imask};
                        if ~exists(obj & tuple)
                            tuple.mask = fmask;
                            tuple.ret_idx = keyI.ret_idx;
                            insert(obj,tuple)
                        end
                    end
                end
            end
            
        end
        
        function [fmasks, fields] = splitContiguousMask(~, key, ref_mask)
            
            % fetch images
            if strcmp(fetch1(experiment.Session & key,'rig'),'2P4')
                [x_pos, y_pos, fieldWidths, fieldHeights, fieldWidthsInMicrons,keys] = ...
                    fetchn(meso.ScanInfoField * meso.SummaryImagesAverage & key,...
                    'x','y','px_width','px_height','um_width');
                
                % calculate initial scale
                pxpitch = mean(fieldWidths.\fieldWidthsInMicrons);
                
                % convert center coordinates to 0,0 coordinates
                x_pos = x_pos - fieldWidths * pxpitch / 2;
                y_pos = y_pos - fieldHeights * pxpitch / 2;
                
                % start indexes
                XX = (x_pos - min(x_pos))/pxpitch;
                YY = (y_pos - min(y_pos))/pxpitch;
                
                % deconstruct the big field of view
                for ifield = 1:length(x_pos)
                    fields(ifield) = keys(ifield).field;
                    fmasks{ifield} = ref_mask(YY(ifield)+1:fieldHeights(ifield)+YY(ifield),...
                        XX(ifield)+1:fieldWidths(ifield)+XX(ifield));
                end
            else % for all other scans there is no need to split the mask
                keys = fetch(meso.ScanInfoField * reso.SummaryImagesAverage & key);
                for ikey = 1:length(keys)
                    fields(ikey) = keys(ikey).field;
                    fmasks{ikey} = ref_mask;
                end
            end
        end
        
        function [area_map, keys, background] = getContiguousMask(obj, key, override)
            
            % fetch masks & keys
            [masks, keys] = fetchn(obj & key,'mask');
            
            % get information from the scans depending on the setup
            if (nargin<3 || ~override) && (strcmp(fetch1(experiment.Session & key,'rig'),'2P4') || length(masks)<2)
                [x_pos, y_pos, fieldWidths, fieldHeights, fieldWidthsInMicrons, masks, areas, avg_image] = ...
                    fetchn(obj * meso.ScanInfoField * meso.SummaryImagesAverage & key,...
                    'x','y','px_width','px_height','um_width','mask','brain_area','average_image');
                
                % calculate initial scale
                pxpitch = mean(fieldWidths.\fieldWidthsInMicrons);
                
                % convert center coordinates to 0,0 coordinates
                x_pos = x_pos - fieldWidths * pxpitch / 2;
                y_pos = y_pos - fieldHeights * pxpitch / 2;
                
                % construct a big field of view
                x_pos = (x_pos - min(x_pos))/pxpitch;
                y_pos = (y_pos - min(y_pos))/pxpitch;
                area_map = zeros(ceil(max(y_pos+fieldHeights)),ceil(max(x_pos+fieldWidths)));
                background = zeros(size(area_map));
                for islice =length(masks):-1:1
                    mask = double(masks{islice})*find(strcmp(areas{islice},unique(areas)));
                    y_idx = ceil(y_pos(islice)+1):ceil(y_pos(islice))+size(mask,1);
                    x_idx = ceil(x_pos(islice)+1):ceil(x_pos(islice))+size(mask,2);
                    back = area_map(y_idx, x_idx);
                    area_map(y_idx, x_idx) = max(cat(3,mask,back),[],3);
                    background(y_idx, x_idx) = avg_image{islice}(1:size(mask,1),1:size(mask,2));
                end
                
            else
                area_map = zeros(size(masks{1}));
                for imasks = 1:length(masks)
                    area_map(masks{imasks}) = imasks;
                end
                background = [];
            end
        end
        
        function plot(obj, varargin)
            
            params.back_idx = [];
            params.bcontrast = 0.4;
            params.contrast = 1;
            params.exp = 1;
            params.sat = 1;
            params.colors = [];
            params.linewidth = 1;
            params.fill = 1;
            params.restrict = [];
            params.red = 1;
            params.fontsize = 12;
            params.fontcolor = [0.4 0 0];
            params.vcontrast = 1;
            
            params = ne7.mat.getParams(params,varargin);
            
            if strcmp(fetch1(proj(experiment.Scan,'aim') & obj,'aim'),'widefield')
                contiguous = 1;
            else
                contiguous = 0;
            end
            
            % get masks
            [area_map, keys, mask_background] = getContiguousMask(obj,fetch(obj),contiguous);
            areas = {keys(:).brain_area}';
            
            % get maps
            if exists(map.RetMap & (map.RetMapScan &  obj))
                background = getBackground(map.RetMap & (map.RetMapScan &  obj));
%                 for iback = 1:size(background,4);
%                    background(:,:,:,iback) = imadjust(background(:,:,:,iback),[0 .1 0.2; .65 .65 .65],[0 0 0; 1 1 1]); 
%                 end
                % if FieldCoordinates exists add it to the background
                if exists(anatomy.FieldCoordinates & proj(anatomy.RefMap & obj) & params.restrict)
                    background = cat(4,background,plot(anatomy.FieldCoordinates & ...
                        proj(anatomy.RefMap & obj) & params.restrict,params));
                    if isempty(params.back_idx)
                        params.back_idx = size(background,4);
                    end
                end
            else
                background = mask_background;
            end
            
            % adjust background contrast
            background = ne7.mat.normalize(abs(background.^ params.bcontrast));
            
            % merge masks with background
            figure
            sat = background(:,:,1,1)*params.sat;
              sat(area_map==0) = 0;
%             im = sat;
%           
%             if nargin<2 || isempty(params.back_idx) || params.back_idx > size(background,4)
%                 image((im));
%             else
%                  if params.fill
%                     imshowpair(im,background(:,:,:,params.back_idx)*params.contrast,'blend')
%                  else
%                     imshow(background(:,:,:,params.back_idx))
%                  end
%             end
im = hsv2rgb(cat(3,ne7.mat.normalize(area_map),sat,background(:,:,1,1)));
            if nargin<2 || isempty(params.back_idx) || params.back_idx > size(background,4)
                image((im));
            else
                imshowpair(im,background(:,:,:,params.back_idx)*params.contrast,'blend')
            end
            hold on
            axis image;
            key = fetch(proj(experiment.Scan) & obj);
            set(gcf,'name',sprintf('Animal:%d Session:%d Scan:%d',key.animal_id,key.session,key.scan_idx))
            
            % place area labels
             un_areas = unique(areas);
            if isempty(params.colors)
                params.colors = hsv(length(un_areas));
            elseif size(params.colors,1)==1
                params.colors =  repmat(params.colors,length(un_areas),1);
            end
           
            for iarea = 1:length(un_areas)
                s = regionprops(area_map==iarea,'Centroid');
                if ~params.fill
                    bw = bwboundaries(area_map==iarea);
                    plot(bw{1}(:,2),bw{1}(:,1),'color',params.colors(iarea,:),'linewidth',params.linewidth);
                end
                text(s(1).Centroid(1),s(1).Centroid(2),un_areas{iarea},'color',params.fontcolor,'fontsize',params.fontsize,'rotation',0,...
                    'HorizontalAlignment','center')
            end
        end
        
        
    end
end