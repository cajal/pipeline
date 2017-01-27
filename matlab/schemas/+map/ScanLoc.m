%{
map.ScanLoc(imported) #
-> experiment.Scan
---
x                         : double                        # x coordinate in pixels
y                         : double                        # y coordinate in pixels
vessels=null              : mediumblob                    # vessel map of the imaging window
vessel_map=null              : mediumblob                 # vessel map with 2P centered at the imaging location
%}

classdef ScanLoc < dj.Relvar & dj.AutoPopulate
    
    properties (Constant)
        popRel = experiment.Scan & (mice.Mice & map.OptImageBar) & 'software="scanimage"' & 'aim="2pScan"'
    end
    
    methods(Access=protected)
        
        function makeTuples( obj, key )
            
            vesselkeys = fetch(experiment.Scan & (mice.Mice & key) & ...
                'aim="vessels"' & 'software="imager"' );
            
            vesskeys = fetch(map.OptImageBar & (mice.Mice & key) ...
                & sprintf('scan_idx>%d',vesselkeys(1).scan_idx));
            
            if isempty(vesskeys)
                vesskeys = fetch(map.OptImageBar & (mice.Mice & key) ...
                    & sprintf('scan_idx<%d',vesselkeys(end).scan_idx));
            end
            
            vessels = fetch1(map.OptImageBar & vesskeys(1),'vessels');
            
            f = figure(key.animal_id);
            if isempty(f.Children)
                imagesc(vessels)
                axis image
                axis off
                colormap gray
                mxidx = 0;
            else
                c = f.Children;
                cc = c.Children;
                txts = cc(isprop(cc,'Text'));
                scans = [];
                for i = 1:length(txts)
                    scans(i) = str2num(txts(i).String);
                end
                mxidx = max(scans);
            end
            hold on
            title(sprintf('AnimalID:%d  Session:%d  ScanIdx:%d Area:%s',...
                key.animal_id, key.session, key.scan_idx, fetch1(experiment.Scan & key,'brain_area')));
            [x, y]=ginput(1);
            text(x,y,num2str(mxidx+1),'horizontalalignment','center',...
                'verticalalignment','middle')
            
            disp 'getting the scan vessel map...'
            k = [];
            k.session = key.session;
            k.animal_id = key.animal_id;
            k.site_number = fetch1(experiment.Scan & key,'site_number');
            ves_key = fetch(experiment.Scan & k & 'aim = "vessels"');
            if ~isempty(ves_key)
                reader = preprocess.getGalvoReader(ves_key);
                vessel_map = reader(:,:,:,:,:);
                vessel_map = mean(vessel_map(:,:,:),3);
            else
                vessel_map = [];
            end
            
            % insert
            key.vessel_map = vessel_map;
            key.vessels = vessels;
            key.x = x;
            key.y = y;
            
            % insert
            insert(obj,key)
        end
    end
    
    methods
        
        function plot(obj)
            
            keys = fetch(mice.Mice & obj & mov3d.Decode );
            for ikey = 1:length(keys)
                
                [x,y,scan,sess,area,mi] = fetchn(obj*experiment.Scan*mov3d.Decode ...
                    & keys(ikey) & 'dec_opt=36','x','y','scan_idx','session','brain_area','mi');
                
                [im(:,:,1),im(:,:,2),im(:,:,3)] = plot(map.OptImageBar & experiment.Session & ...
                    keys(ikey) & 'selected=1' & 'axis = "horizontal"');
                im = hsv2rgb(im);
                figure
                imshow(im)
                set(gcf,'name',num2str(keys(ikey).animal_id))
                axis image
                axis off
                colormap gray
                hold on
                for i = 1:length(x)
                    plot(x(i),y(i),'*k')
                    text(x(i)+5,y(i)+5,[num2str(sess(i)) '-' num2str(scan(i)) ...
                        ' (' area{i} ', ' num2str(roundall(mean(mi{i}),0.01)) ')'])
                end
                
            end
        end
        
        function [ramp, keys] = getRelAmp(obj)
            
            keys = fetch(obj & (mice.Mice & ...
                (map.OptImageBar& 'selected=1' & 'axis = "horizontal"')) );
            ramp = nan(length(keys),1);
            for ikey = 1:length(keys)
            
                [x,y] = fetchn(obj*experiment.Scan*mov3d.Decode ...
                    & keys(ikey) & 'dec_opt=36','x','y');

                amp = fetch1(map.OptImageBar & (mice.Mice & ...
                    keys(ikey)) & 'selected=1' & 'axis = "horizontal"','amp');

%                 amp = normalize(amp);

                mask = zeros(size(amp));
                mask(round(x),round(y)) = 1;
                mask = logical(convn(mask,gausswin(5)*gausswin(5)','same'));

                ramp(ikey) = mean(amp(mask));
            end
            
        end
    end
    
end