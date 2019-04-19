%{
# mask center of mass of a segmented cell
-> fuse.ScanSetUnit
-> fuse.ScanSet
ref_idx                 : smallint        # reference map index for each animal
-----
xloc                 : double # x location in microns relative to the center of reference map
yloc                 : double # y location in microns relative to the center of reference map
zloc                 : double # z location in micro meters relative to the surface
%}

classdef UnitCoordinates < dj.Computed
    
    properties (Constant)
        keySource = proj(fuse.ScanSet) * proj(anatomy.FieldCoordinates)
    end
    
    methods (Access=protected)
        
        function makeTuples(self, key)
            
            % get scan info
            [x_off,y_off,pxpitch,depth,tform] = fetch1(anatomy.FieldCoordinates & key,...
                'x_offset','y_offset','pxpitch','field_depth','tform');
            setup = fetch1(experiment.Scan * experiment.Session & key,'rig');
            
            if strcmp(setup,'2P4')
                [px_width, px_height, unit_id] = ...
                    fetchn((meso.ScanSetUnitInfo & (fuse.ScanSetUnit & key)),...
                    'px_x','px_y','unit_id');
                [field_height, field_width]  = fetch1(meso.ScanInfoField & key,'px_height','px_width');
            else
                [px_width, px_height, unit_id] = ...
                    fetchn((reso.ScanSetUnitInfo & (fuse.ScanSetUnit & key)),...
                    'px_x','px_y','unit_id');
                 [field_height, field_width]  = fetch1(reso.ScanInfo & key,'px_height','px_width');
            end
            % correct for x axis mirroring
            tform = transpose(tform);

            for imask = 1:length(unit_id)
                % get mask position
                centerOfMass = [px_width(imask) - field_width/2 px_height(imask) - field_height/2 0];
                
                % calculate correct location
                loc = tform*centerOfMass';
                
                % insert
                key.unit_id = unit_id(imask);
                key.xloc = (loc(1)+x_off)*pxpitch;
                key.yloc = (loc(2)+y_off)*pxpitch;
                key.zloc = depth;
                insert(self,key);
                
            end
        end
    end
    
    methods
        function plot(self,varargin)
            params.markersize = 1;
            params.sigma = 5;
            params.vcontrast = 0.5;
            params.exp = 2;
            params = ne7.mat.getParams(params,varargin);
            
            [x, y, brain_area] = fetchn(self*anatomy.AreaMembership,'xloc','yloc','brain_area');
            un_areas = unique(brain_area);
            ref_key = fetch(anatomy.RefMap & ( proj(anatomy.RefMap) & self),'*');
            colors = hsv(length(un_areas)+1);
            retmap = eval([ref_key.ref_table,'&ref_key']);
            [h,s,v] = plot(retmap,params);
            for imap = 1:retmap.count
                figure
                im(:,:,1) = h{imap};
                im(:,:,2) = s{imap};
                im(:,:,3) = v{imap};
                imagesc(hsv2rgb(im))
                axis off
                hold on
                for iarea = 1:length(un_areas)
                    idx = strcmp(brain_area,un_areas{iarea});
                    plot(x(idx)/ref_key.pxpitch + size(ref_key.ref_map,1)/2,y(idx)/ref_key.pxpitch + size(ref_key.ref_map,2)/2,...
                        '.','color',colors(iarea,:),'markersize',params.markersize)
                end
            end
        end
        
        function plotRepeats(self,varargin)
            params.markersize = 10;
            params.sigma = 5;
            params.vcontrast = 0.5;
            params.exp = 2;
            params = ne7.mat.getParams(params,varargin);
            
            [x, y, keys] = fetchn(self,'xloc','yloc');

            r= nan(length(keys),1);
            ikey= 0;
            for key = keys'
                ikey = ikey+1;
                r(ikey) = nanmean(fetchn(obj.RepeatsUnit & key,'r'));
            end
            r(isnan(r)) = 0;
            colors = ([ones(size(r)) 1-normalize(r) 1-normalize(r)]);
            ref_key = fetch(anatomy.RefMap & ( proj(anatomy.RefMap) & self),'*');
            retmap = eval([ref_key.ref_table,'&ref_key']);
            [h,s,v] = plot(retmap,params);
            for imap = 1:retmap.count
                figure
                im(:,:,1) = h{imap};
                im(:,:,2) = s{imap};
                im(:,:,3) = v{imap};
                imagesc(hsv2rgb(im))
                axis off
                hold on
                for icell = 1:length(r)
                 
                    plot(x(icell)/ref_key.pxpitch + size(ref_key.ref_map,1)/2,y(icell)/ref_key.pxpitch + size(ref_key.ref_map,2)/2,...
                        '.','color',colors(icell,:),'markersize',params.markersize)
                end
            end
            end
        
        function plot3d(self)
            plot3d(anatomy.FieldCoordinates & self)
        end
    end
end