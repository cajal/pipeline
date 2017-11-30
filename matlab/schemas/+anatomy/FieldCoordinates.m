%{
# Relative scan coordinates to a reference map
-> experiment.Scan
field                   : tinyint         # field #
---
x_offset                : double          # x center coordinate in pixels
y_offset                : double          # y center coordinate in pixels
field_depth             : double          # depth of slice from surface in microns
tform                   : mediumblob      # transformation matrix for rotation,scale,flip relative to vessel map
pxpitch                 : double          # estimated pixel pitch of the reference map (px)
ref_key                 : mediumblob      # key of the reference vessel map
ref_table               : varchar         # reference table
ref_map                 : mediumblob      # reference vessel map
%}

classdef FieldCoordinates < dj.Imported
    
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            insert( obj, key );
        end
    end
    
    methods
        function alignScan(self,keyI,REF,varargin)
            
            params.global_rotation = -30; % rotation of the ref_map 
            params.ref_height = 3530;     % (microns) 
            
            params = getParams(params,varargin);
            
            ref_map = [];
            ref_key = [];
            ref_table = [];
            
            % handle image ref_map
            if ismatrix(REF) && numel(REF)>4
                ref_map = REF;
            else
                ref_table = REF.className;
            end
            
            % get map from coresponding table
            switch ref_table
                case 'map.OptImageBar'
                    % get the intrinsic vessel map
                    [ref_map] = fetch1(REF,'vessels');
                case 'meso.SummaryImagesAverage'
                    ref_map = fetch(REF,'average_image');
            end
            assert(~isempty(ref_map),'No maps found!'); 
               
            % get the scamimage reader
            [setup, depth] = fetch1(experiment.Scan * experiment.Session & keyI,...
                'rig','depth');
            
            % init tform params, specific to each setup
            tfp = struct('scale',1,'rotation',90,'fliplr',0,'flipud',0);
            
            % get information from the scans depending on the setup
            if strcmp(setup,'2P4')   
                [x_pos, y_pos, slice_pos, fieldWidths, fieldHeights, fieldWidthsInMicrons, frames, field_num] = ...
                    fetchn(meso.ScanInfoField * meso.SummaryImagesAverage & keyI & 'channel = 1',...
                    'x','y','z','px_width','px_height','um_width','average_image','field');
                x_pos = -x_pos;
            else
                tfp.fliplr = 1;
                [fieldWidths, fieldHeights, fieldWidthsInMicrons, frames, slice_pos, field_num] = ...
                    fetchn(reso.ScanInfo * reso.SummaryImagesAverage & keyI & 'channel = 1',...
                    'px_width','px_height','um_width','average_image','z','field');
                x_pos = zeros(size(fieldWidths));y_pos = x_pos;
            end
            
            % calculate initial scale
            pxpitch = mean(fieldWidths.\fieldWidthsInMicrons);
            
            % construct the big mesoscope field of view
            x_pos = x_pos/pxpitch;
            y_pos = y_pos/pxpitch;
            x_pos = x_pos + abs(min(x_pos))+1;
            y_pos = y_pos + abs(min(y_pos))+1;
            im = zeros(ceil(max(x_pos+fieldWidths)),ceil(max(y_pos+fieldHeights)));
            for islice =length(frames):-1:1
                frame = frames{islice};
                frame(frame>prctile(frame(:),99)) = prctile(frame(:),99);
                frame = self.filterImage(self.normalize(frame),self.createTform(tfp));
                im(ceil(x_pos(islice)):ceil(x_pos(islice))+size(frame,1)-1, ceil(y_pos(islice)):ceil(y_pos(islice))+size(frame,2)-1) = frame;
            end
            
            % Align scans
            [x_offset, y_offset, rotation, tfp.scale, go] = ...
                self.alignImages(self.normalize(ref_map),self.normalize(im),...
                'scale',pxpitch/(params.ref_height/size(ref_map,1)),'rotation',params.global_rotation);
            tfp.rotation = tfp.rotation + rotation;
            
            % Insert overlaping masks
            if go
                for islice = 1:length(frames)
                    x = size(ref_map,1)/2 - x_offset - size(frames{islice},2)*tfp.scale/2 - x_pos(islice)*tfp.scale+1;
                    y = size(ref_map,2)/2 - y_offset - size(frames{islice},1)*tfp.scale/2 - y_pos(islice)*tfp.scale+1;
                    theta = abs(atand(y/x)) + 180;
                    if sign(x)>=0 && sign(y)<0
                        theta = 360 - theta;
                    elseif sign(x)<0 && sign(y)<0
                        theta = 180 + theta;
                    elseif sign(x)<0 && sign(y)>=0
                        theta = 180 - theta;
                    end
                    phi = theta + rotation;
                    hyp = sqrt(x^2 + y^2);
                    tuple = keyI;
                    tuple.field = field_num(islice);
                    tuple.x_offset = cosd(phi)*hyp;
                    tuple.y_offset = sind(phi)*hyp;
                    tuple.tform = self.createTform(tfp);
                    tuple.pxpitch = pxpitch/tfp.scale; % estimated pixel pitch of the vessel map;
                    tuple.ref_key = ref_key;
                    tuple.field_depth = slice_pos(islice) - depth;
                    tuple.ref_table = ref_table;
                    tuple.ref_map = ref_map;
                    makeTuples(anatomy.FieldCoordinates,tuple)
                end
            else
                disp 'Exiting...'
            end
        end
        
        function plot(self)
            
            % get the ref_map
            ref_map = fetch1(self & fuse.ScanDone,'ref_map');
    
            
             % get the setup
            setup = fetch1(experiment.Session & self, 'rig');
            
            % fetch images
            if strcmp(setup,'2P4')   
             	[frames,x,y,tforms] = fetchn(meso.SummaryImagesAverage * self & 'channel = 1',...
                'average_image','x_offset','y_offset','tform');
            else
            	[frames,x,y,tforms] = fetchn(reso.SummaryImagesAverage * self & 'channel = 1',...
                'average_image','x_offset','y_offset','tform');
            end
        
            % plot
            figure
            im = zeros(size(ref_map,1),size(ref_map,2),3);
            im(:,:,1) = self.normalize(ref_map);
            for islice = 1:length(frames)
                
                x_offset = x(islice);
                y_offset = y(islice);
                imS = self.filterImage(self.normalize(frames{islice}),tforms{islice});
                YY = round(y_offset + size(ref_map,2)/2 - size(imS,2)/2);
                XX = round(x_offset + size(ref_map,1)/2 - size(imS,1)/2);
                
                im(XX:size(imS,1)+XX-1,YY:size(imS,2)+YY-1,2) = ...
                    im(XX:size(imS,1)+XX-1,YY:size(imS,2)+YY-1,2) + self.normalize(abs(imS.^.7));
            end
            image(im)
            axis image
            axis off
        end
        
        function plot3D(self)
            
            % get the ref_map
            ref_map = fetch1(self & fuse.ScanDone,'ref_map');
            
             % get the setup
            setup = fetch1(experiment.Session & self, 'rig');
            
            % fetch images
            if strcmp(setup,'2P4')   
             	[frames,x,y,tforms,pxpitch] = fetchn(meso.SummaryImagesAverage * self & 'channel = 1',...
                'average_image','x_offset','y_offset','tform','pxpitch');
            else
            	[frames,x,y,tforms,pxpitch] = fetchn(reso.SummaryImagesAverage * self & 'channel = 1',...
                'average_image','x_offset','y_offset','tform','pxpitch');
            end
            
            % plot
            figure
            ref_map = self.normalize(ref_map*2.^2)*0.5;
            ref_map = ref_map*50;
            self.image3D(0,0,0,ref_map, abs(ref_map-max(ref_map(:)))*2,pxpitch(1));
            for islice = 1:length(frames)
                x_offset = x(islice);
                y_offset = y(islice);
                imS = self.filterImage(self.normalize(frames{islice}),tforms{islice});
                imS = self.normalize(abs(imS.^.7));
                imS2 = imS*50+50;
                self.image3D(y_offset,-x_offset,-depth(islice)/pxpitch(islice),imS2,imS*100,pxpitch(islice));
            end
            axis image
            cmap = bsxfun(@times,[1 0 0],[0:0.02:1]');  
            cmap2 =  bsxfun(@times,[0 1 0],[0:0.02:1]');
            colormap([cmap;cmap2])
        end
        
        function fmask = filterMask(self,mask)
            
             % get the setup
            setup = fetch1(experiment.Session & self, 'rig');
            
            % fetch images
            if strcmp(setup,'2P4')   
             	[frame,x_offset,y_offset,tform] = fetch1(meso.SummaryImagesAverage * self & 'channel = 1',...
                'average_image','x_offset','y_offset','tform');
            else
            	[frame,x_offset,y_offset,tform] = fetch1(reso.SummaryImagesAverage * self & 'channel = 1',...
                'average_image','x_offset','y_offset','tform');
            end
            
            sz = size(frame);
%             imS = self.filterImage(self.normalize(frame),tform);
            imS = frame;
%             YY = round(y_offset + size(mask,1)/2 - size(imS,2)/2); % changed from ize(imS,1)/2)
%             XX = round(x_offset + size(mask,2)/2 - size(imS,1)/2); % changed from ize(imS,2)/2)
            YY = round(-x_offset + size(mask,1)/2 - size(imS,2)/2)+1; % changed from ize(imS,1)/2)
            XX = round(y_offset + size(mask,2)/2 - size(imS,1)/2)+1; % changed from ize(imS,2)/2)
            fmask = mask(XX:size(imS,1)+XX-1,YY:size(imS,2)+YY-1);
%             fmask = self.filterImage(self.normalize(fmask),tform,1)>0;
            fmask = fmask(...
                round(size(fmask,1)/2)-floor(sz(1)/2)+1:round(size(fmask,1)/2)+floor(sz(1)/2),...
                round(size(fmask,2)/2)-floor(sz(2)/2)+1:round(size(fmask,2)/2)+floor(sz(2)/2));
            
%             fmask = fmask(...
%                 round(size(fmask,1)/2)-floor(sz(1)/2):round(size(fmask,1)/2)+floor(sz(1)/2)+1,...
%                 round(size(fmask,2)/2)-floor(sz(2)/2):round(size(fmask,2)/2)+floor(sz(2)/2)+1);
        end
    end
    
    methods (Static)
        function x = normalize(x)
            if length(unique(x(:)))>1
                x = (x - nanmin(x(:)))/(nanmax(x(:)) - nanmin(x(:)));
            end
        end
        
        function [x,y,rot,scale,go] = alignImages(image1,image2,varargin)
            
            % set initial parameters
            params.rotation = 0;
            params.scale = 0;
            params.x = 0;
            params.y = 0;
            params.resize = 0.5;
            params = getParams(params,varargin);
            
            if params.x == 0 && params.y == 0
                params.x = round(size(image1,1)/2 - size(image2,1)*params.scale/2);
                params.y = round(size(image1,2)/2 - size(image2,2)*params.scale/2);
            end
            
            rot = params.rotation;
            scale = params.scale;
            x = params.x * params.resize;
            y = params.y * params.resize;
            fine = 1;
            
            % assign images
            vessels = imresize(normalize(image1),params.resize);
            im1 = vessels;
            im2 = imresize(normalize(image2),params.resize);
            hf = figure('NumberTitle','off','Menubar','none','Name','Align Images','KeyPressFcn',@eval_input);
            f_pos = get(hf,'outerposition');
            imh = image(im2); axis image; axis off
            redraw
            disp 'Align the images'
            
            % wait until done
            go = false;
            while ~go
                try if ~ishandle(hf);break;end;catch;break;end
                pause(0.2);
            end
            
            % adjust x,y
            x = x/params.resize;
            y = y/params.resize;
            
            function eval_input(~, event)
                global temp
                switch event.Key
                    case 'shift'
                        if fine==1
                            fine=0.5;
                        else
                            fine = 1;
                        end
                    case 'downarrow'
                        if x<size(im1,1)/2
                            x = x+1*fine;
                            redraw
                        end
                    case 'uparrow'
                        if x>1
                            x = x-1*fine;
                            redraw
                        end
                    case 'rightarrow'
                        if y<size(im1,2)/2
                            y = y+1*fine;
                            redraw
                        end
                    case 'leftarrow'
                        if y>1
                            y = y-1*fine;
                            redraw
                        end
                    case 'comma'
                        rot = rot+0.5*fine;
                        redraw
                    case 'period'
                        rot = rot-0.5*fine;
                        redraw
                    case 'equal'
                        if scale<1
                            scale = scale+0.005*fine;
                            redraw
                        end
                    case 'hyphen'
                        if scale>0
                            scale = scale-0.005*fine;
                            redraw
                        end
                    case 'return'
                        go = true;
                        close(gcf)
                    case 'escape'
                        close(gcf)
                    case 'space'
                        if all(im2(:)==0)
                            im2 = temp;
                        else
                            temp = im2;
                            im2 = zeros(size(im2));
                        end
                        redraw
                    case 'f' % toggle fullscreen
                        set(hf,'units','normalized')
                        p = get(hf,'outerposition');
                        if all(p~=[0 0 1 1])
                            set(hf,'outerposition',[0 0 1 1]);
                        else
                            set(hf,'outerposition',f_pos);
                        end
                        set(hf,'units','pixels')
                end
            end
            
            function redraw
                % draw image with masks
                im1 = (imrotate(imresize(vessels,1/scale),-rot,'crop'));
                im3 = zeros(size(im1,1),size(im1,2),3);
                im3(:,:,1) = im1;
                im3(round(x/scale):size(im2,1)+round(x/scale) - 1,...
                    round(y/scale):size(im2,2)+round(y/scale) - 1,2) = im2;
                set(gcf,'name',['X:' num2str(x) ' Y:' num2str(y) 'rot:' num2str(rot) ' scale:' num2str(scale)])
                imh.CData = (im3);
            end
        end
        
        function tform = createTform(tfp)
            theta = tfp.rotation/180*pi; % convert to radians
            R = [cos(theta) -sin(theta) 0; ...
                sin(theta) cos(theta) 0;...
                0 0 1];
            tform = R*diag([tfp.scale*(1-2*tfp.fliplr),...
                tfp.scale*(1-2*tfp.flipud),...
                (1-2*tfp.fliplr)*(1-2*tfp.flipud)]);
        end
        
        function fim = filterImage(im,tform,inv)
            tform = projective2d(tform([5 4 7;2 1 8;3 6 9])); % correct for flipped image axis in matlab
            if nargin>2 && inv
                tform = invert(tform);
            end
            fim = imwarp(im,tform);
        end
        
        function image3D(x,y,z,I,IA,scale)
            I = flipud(I);
            IA = flipud(IA);
            
            if nargin<6; scale = 1; end

            %# coordinates
            [X,Y] = meshgrid(1:size(I,2), 1:size(I,1));
            Z = ones(size(I,1),size(I,2))*z;
            X = X - size(I,2)/2 + x;
            Y = Y - size(I,1)/2 + y;
            %# plot each slice as a texture-mapped surface (stacked along the Z-dimension)
            surface('XData',X*scale, 'YData',Y*scale, 'ZData',Z*scale, ...
                'CData',I, 'CDataMapping','direct', ...
                'EdgeColor','none', 'FaceColor','texturemap',...
                'AlphaData',IA,'FaceAlpha','texturemap','alphadatamapping','direct')


        end
    end
    
end