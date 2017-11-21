%{
# mask center of mass of a segmented cell
-> fuse.ScanSetUnit
-----
xloc                 : double # x location in microns relative to the center of reference map
yloc                 : double # y location in microns relative to the center of reference map
zloc                 : double # z location in micro meters relative to the surface
%}

classdef UnitCoordinates < dj.Computed
    
    properties
        keySource = anatomy.FieldCoordinates * fuse.ScanSet
    end
    
    methods (Access=protected)
        
        function makeTuples(self, key)
            
            % get scan info
            [x_off,y_off,pxpitch,depth,tform] = fetch1(anatomy.FieldCoordinates & key,...
                'x_offset','y_offset','pxpitch','depth','tform');
            setup = fetch1(experiment.Scan * experiment.Session & key,'rig');
            
            if strcmp(setup,'2P4')
                [px_width, px_height, unit_id, field_height, field_width] = ...
                    fetchn((meso.ScanSetUnitInfo & (fuse.ScanSetUnit & key)) * proj(meso.ScanInfoField,'px_height','px_width'),...
                    'px_x','px_y','unit_id','px_height','px_width');
            else
                [px_width, px_height, unit_id, field_height, field_width] = ...
                    fetchn((reso.ScanSetUnitInfo & (fuse.ScanSetUnit & key)) * proj(reso.ScanInfo,'px_height','px_width'),...
                    'px_x','px_y','unit_id','px_height','px_width');
            end
            
            for imask = 1:length(unit_id)
                % get mask position
                centerOfMass = [px_width(imask) - field_width/2 px_height(imask) - field_height/2 0];
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
        function plot3d(self)
            plot3d(anatomy.FieldCoordinates & self)
        end
    end
end