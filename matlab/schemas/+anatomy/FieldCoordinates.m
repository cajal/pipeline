%{
# Relative scan coordinates to a reference map
-> experiment.Scan
-> anatomy.RefMap
field                   : tinyint         # field #
---
x_offset                : double          # x center coordinate in pixels
y_offset                : double          # y center coordinate in pixels
field_depth             : double          # depth of slice from surface in microns
tform                   : mediumblob      # transformation matrix for rotation,scale,flip relative to vessel map
pxpitch                 : double          # estimated pixel pitch of the reference map (microns per pixel)
%}

classdef FieldCoordinates < dj.Manual
    methods
        function alignScan(self,keyI, varargin)

            params.contrast = 1;
            params.global_rotation =[]; % rotation of the ref_map
            params.scan_rotation = 0;
            params.x_offset = [];
            params.y_offset = [];
            params.scale = [];
            params.figure = [];
            params.amp = 1;
            params.noref = false;

            params = ne7.mat.getParams(params,varargin);

            % init tform params, specific to each setup
            tfp = struct('scale',1,'rotation',params.scan_rotation,'fliplr',0,'flipud',0);

             % get the scamimage reader
            [setup, depth] = fetch1(experiment.Scan * experiment.Session & keyI,...
                'rig','depth');

             % get information from the scans depending on the setup
            if strcmp(setup,'2P4')
                [x_pos, y_pos, slice_pos, fieldWidths, fieldHeights, fieldWidthsInMicrons, frames, field_num] = ...
                    fetchn(meso.ScanInfoField * meso.SummaryImagesAverage & keyI & 'channel = 1',...
                    'x','y','z','px_width','px_height','um_width','average_image','field');
            else
                tfp.fliplr = 1;
                tfp.flipud = 1;
                [fieldWidths, fieldHeights, fieldWidthsInMicrons, frames, slice_pos, field_num] = ...
                    fetchn(reso.ScanInfo * reso.SummaryImagesAverage * reso.ScanInfoField & keyI & fuse.ScanSet,...
                    'px_width','px_height','um_width','average_image','z','field');
                depth = 0; % because this is included in slice_pos of the reso
                x_pos = zeros(size(fieldWidths));y_pos = x_pos;
            end

            % if no input is given then scan coordinates are going to be used
            if params.noref
                ref_key = createRef(anatomy.RefMap,experiment.Scan & keyI);
                for islice = 1:length(frames)
                    tuple = keyI;
                    tuple.ref_idx = ref_key.ref_idx;
                    tuple.field = field_num(islice);
                    tuple.x_offset = 0;
                    tuple.y_offset = 0;
                    tuple.tform = 1;
                    tuple.pxpitch = 1; % estimated pixel pitch of the vessel map;
                    tuple.field_depth = slice_pos(islice) - depth;
                    insert(self,tuple)
                end
                return
            end

            % get reference map information
            ref_key = fetch(proj(anatomy.RefMap) & keyI);
            if ~exists(anatomy.RefMap & ref_key)
                ref_key = createRef(anatomy.RefMap,fetch(mice.Mice & keyI));
            end

            if isfield(keyI,'ret_idx');keyI = rmfield(keyI,'ret_idx');end
            [ref_map, ref_pxpitch, software] = fetch1(anatomy.RefMap * experiment.Scan  & ref_key,...
                'ref_map','pxpitch','software');
            switch software
                case 'imager'
                    if isempty(params.global_rotation)
                        params.global_rotation = 60; % rotation of the ref_map
                    end
                case 'scanimage'
                    if isempty(params.global_rotation)
                        params.global_rotation = 0; % rotation of the ref_map
                    end
            end

            % calculate initial scale
            pxpitch = mean(fieldWidths.\fieldWidthsInMicrons);

            if length(frames)>1
                % convert center coordinates to 0,0 coordinates
                x_pos = x_pos - fieldWidths * pxpitch / 2;
                y_pos = y_pos - fieldHeights * pxpitch / 2;
                
                % shift and convert microns to pixels
                x_pos = (x_pos - min(x_pos))/pxpitch;
                y_pos = (y_pos - min(y_pos))/pxpitch;
                
                % construct a big field of view
                im = zeros(ceil(max(y_pos+fieldHeights)),ceil(max(x_pos+fieldWidths)));
                for islice =length(frames):-1:1
                    frame = self.filterImage(ne7.mat.normalize(frames{islice}),self.createTform(tfp));
                    im(ceil(y_pos(islice)+1):ceil(y_pos(islice))+size(frame,1), ...
                        ceil(x_pos(islice)+1):ceil(x_pos(islice))+size(frame,2)) = ...
                        self.processImage(frame,'exp',params.contrast);
                end
            else
                im = ne7.mat.normalize(frames{1}.^params.contrast);
                x_pos = 0;
                y_pos = 0;
            end

            if isempty(params.scale)
                params.scale = pxpitch/ref_pxpitch;
            end

            % Align scans
            [x_offset, y_offset, rotation, tfp.scale, go] = ...
                self.alignImages(ne7.mat.normalize(abs(double(ref_map).^params.amp)),ne7.mat.normalize(im),...
                'scale',params.scale,'rotation',params.global_rotation,'x',params.x_offset,'y',params.y_offset,'figure',params.figure);
            tfp.rotation = tfp.rotation + rotation;

            % Insert overlaping masks
            if go
                for islice = 1:length(frames)
                    x = size(ref_map,2)/2 - y_offset - size(frames{islice},2)*tfp.scale/2 - x_pos(islice)*tfp.scale;
                    y = size(ref_map,1)/2 - x_offset - size(frames{islice},1)*tfp.scale/2 - y_pos(islice)*tfp.scale;
                    if x==0;theta = 180;else; theta = abs(atand(y/x)) + 180;end
                    if sign(x)>=0 && sign(y)<0
                        theta = 360 - theta;
                    elseif sign(x)<0 && sign(y)<0
                        theta = 180 + theta;
                    elseif sign(x)<0 && sign(y)>=0
                        theta = 180 - theta;
                    end
                    phi = theta -rotation;
                    hyp = sqrt(x^2 + y^2);
                    tuple = keyI;
                    tuple.field = field_num(islice);
                    tuple.x_offset = cosd(phi)*hyp;
                    tuple.y_offset = sind(phi)*hyp;
                    tuple.tform = self.createTform(tfp);
                    tuple.pxpitch = pxpitch/tfp.scale; % estimated pixel pitch of the vessel map;
                    tuple.field_depth = slice_pos(islice) - depth;
                    insert(self,tuple)
                end
            else
                disp 'Exiting...'
            end
        end

        function out_im = plot(self,varargin)

            params.exp = 1;
            params.inv = 0;
            params.vcontrast = 1;
            params.red = 1;
            params = ne7.mat.getParams(params,varargin);

            assert(exists(self),'No fields found!')

            % get the ref_map
            ref_map = fetchn(proj(anatomy.RefMap,'ref_map') & self,'ref_map');
            ref_map = single(ref_map{1});

            % get the setup
            setups = fetchn(experiment.Session & self, 'rig');
            setup = setups{1};

            % fetch images
            if strcmp(setup,'2P4')
                [frames1,x,y,tforms] = fetchn(meso.SummaryImagesAverage * self ,...
                    'average_image','x_offset','y_offset','tform');

                  [frames2,x,y,tforms] = fetchn(meso.SummaryImagesCorrelation * self ,...
                    'correlation_image','x_offset','y_offset','tform');
                for iframe = 1:length(frames1);
                    frames{iframe} = frames1{iframe}.*frames2{iframe};
                end
            else
                [frames,x,y,tforms,keys] = fetchn(reso.SummaryImagesAverage * self & fuse.ScanSet,...
                    'average_image','x_offset','y_offset','tform');
            end

            % apply field transformations
            ref_map = ne7.mat.normalize(abs(ref_map.^params.vcontrast));
            if params.inv; ref_map = 1-ref_map;end
            idxX = [];idxY = []; frame = [];
            for islice = 1:length(frames)
                x_offset = x(islice);
                y_offset = y(islice);
                imS = self.filterImage(ne7.mat.normalize(frames{islice}),tforms{islice});
                YY = round(y_offset + size(ref_map,1)/2 - size(imS,1)/2)+1;
                XX = round(x_offset + size(ref_map,2)/2 - size(imS,2)/2)+1;
                idxX{islice} = XX:size(imS,2)+XX-1;
                idxY{islice} = YY:size(imS,1)+YY-1;
                frame{islice} = self.processImage(imS,'exp',params.exp);
            end

            % find negetive indexes and expand mask
            y_range = range([0 size(ref_map,1) cellfun(@min,idxY) cellfun(@max,idxY)])+1;
            x_range = range([0 size(ref_map,2) cellfun(@min,idxX) cellfun(@max,idxX)])+1;
            mnX = abs(min([0 cellfun(@min,idxX)]))+1;
            mnY = abs(min([0 cellfun(@min,idxY)]))+1;
            idxX = cellfun(@(x) x+mnX,idxX,'uni',0);
            idxY = cellfun(@(x) x+mnY,idxY,'uni',0);

            % construct image
            im = zeros(y_range,x_range,3);
             im(1+mnY:size(ref_map,1)+mnY,1+mnX:size(ref_map,2)+mnX,1) = ref_map;

            % put frames
            for islice = 1:length(frames)
                im(idxY{islice},idxX{islice},2) = ...
                    (im(idxY{islice},idxX{islice},2) + frame{islice});
            end
            im(:,:,2) = ne7.mat.normalize(im(:,:,2));
            im = im(mnY+(1:size(ref_map,1)),mnX+(1:size(ref_map,2)),:);

            if ~params.red
                A = repmat(im(:,:,1),1,1,3);
                A(:,:,1) = (im(:,:,2)>0)*.3;
                A(:,:,2) = normalize(im(:,:,2).^1.5);
                im = hsv2rgb(A);
            end
            
            % plot
            if ~nargout
                figure
                image((im))
                axis image
                axis off
            else
%                 im(:,:,3) = (im(:,:,2)>0)*0.33;
%                 out_im = hsv2rgb(im(:,:,[3 2 1]));
                out_im = im;
            end
        end

        function plot3D(self)

            % get the ref_map
            ref_map = fetchn(proj(anatomy.RefMap,'ref_map') & self,'ref_map');

            % fetch images
            setup = fetchn(experiment.Session & self, 'rig');
            if strcmp(setup{1},'2P4')
                [frames,x,y,tforms,pxpitch,depth] = fetchn(meso.SummaryImagesAverage * self & fuse.ScanSet,...
                    'average_image','x_offset','y_offset','tform','pxpitch','field_depth');
            else
                [frames,x,y,tforms,pxpitch,depth] = fetchn(reso.SummaryImagesAverage * self & fuse.ScanSet,...
                    'average_image','x_offset','y_offset','tform','pxpitch','field_depth');
            end

            % plot
            figure
            ref_map = ne7.mat.normalize(ref_map{1}*2.^2)*0.5;
            ref_map = ref_map*50;
            self.image3D(0,0,0,ref_map, abs(ref_map-max(ref_map(:)))*2,pxpitch(1));
            for islice = 1:length(frames)
                imS = self.filterImage(ne7.mat.normalize(frames{islice}),tforms{islice});
                imS = self.processImage(imS,'exp',0.7);
                imS2 = imS*50+50;
                self.image3D(x(islice),y(islice),-depth(islice)/pxpitch(islice),imS2,imS*100,pxpitch(islice));
            end
            axis image
            cmap = bsxfun(@times,[1 0 0],[0:0.02:1]');
            cmap2 =  bsxfun(@times,[0 1 0],[0:0.02:1]');
            colormap([cmap;cmap2])
        end

        function fmask = filterMask(self, ref_mask)

            % fetch images
            setup = fetch1(experiment.Session & self, 'rig');
            if strcmp(setup,'2P4')
                [frame,x_offset,y_offset,tform] = fetch1(meso.SummaryImagesAverage * self & 'channel = 1',...
                    'average_image','x_offset','y_offset','tform');
            else
                [frame,x_offset,y_offset,tform] = fetch1(reso.SummaryImagesAverage * self & fuse.ScanSet,...
                    'average_image','x_offset','y_offset','tform');
            end

            sz = size(frame);
            ref_mask = padarray(ref_mask,[100 100]); % pad image in case the ref_mask is not large enough
            imS = self.filterImage(ne7.mat.normalize(frame),tform);            % apply rotation/flips
            YY = round(y_offset + size(ref_mask,1)/2 - size(imS,1)/2); % convert center coordinates to 0,0 coordinates
            XX = round(x_offset + size(ref_mask,2)/2 - size(imS,2)/2); % convert center coordinates to 0,0 coordinates
            fmask = ref_mask(YY+1:size(imS,1)+YY,XX+1:size(imS,2)+XX);
            fmask = self.filterImage(fmask,tform,1);
            fmask = fmask(...
                round(size(fmask,1)/2)-floor(sz(1)/2)+1:floor(size(fmask,1)/2)+floor(sz(1)/2),...
                round(size(fmask,2)/2)-floor(sz(2)/2)+1:floor(size(fmask,2)/2)+floor(sz(2)/2));
        end
    end

    methods (Static)
        function [x,y,rot,scale,go] = alignImages(image1,image2,varargin)

            % set initial parameters
            params.rotation = 0;
            params.scale = 1;
            params.x = [];
            params.y = [];
            params.resize = .5;
            params.figure = [];

            params = ne7.mat.getParams(params,varargin);

            if isempty(params.x) || isempty(params.y)
                params.x = max([0 round(size(image1,1)/2 - size(image2,1)*params.scale/2)]);
                params.y = max([0 round(size(image1,2)/2 - size(image2,2)*params.scale/2)]);
            end

            rot = params.rotation;
            scale = params.scale;
            x = params.x * params.resize;
            y = params.y * params.resize;
            fine = 1;

            % assign images
            vessels = imresize(ne7.mat.normalize(image1),params.resize);
            im1 = vessels;
            im2 = imresize(ne7.mat.normalize(image2),params.resize);
            if isempty(params.figure); hf = figure; else hf = params.figure;end
            set(hf,'NumberTitle','off','Menubar','none','Name','Align Images','KeyPressFcn',@eval_input);
            f_pos = get(hf,'outerposition');
            imh = image(im2); axis image; axis off
            redraw
            disp 'Align the images'

            % wait until done
            go = false;esc = false;
            while ~go && ~esc
                try if ~ishandle(hf);break;end;catch;break;end
                pause(0.2);
            end

            if isempty(params.figure)
                close(hf)
            else
                clf(hf,'reset')
            end

            % adjust x,y
            x = x/params.resize;
            y = y/params.resize;

            function eval_input(~, event)
                global temp
                switch event.Key
                    case 'shift'
                        if fine==4
                            fine=0.5;
                        else
                            fine = 4;
                        end
                    case 'downarrow'
                        if x<size(im1,1)/2
                            x = x+1*fine;
                            redraw
                        end
                    case 'uparrow'
%                         if x>0
                            x = x-1*fine;
                            redraw
%                         end
                    case 'rightarrow'
                        if y<size(im1,2)/2
                            y = y+1*fine;
                            redraw
                        end
                    case 'leftarrow'
%                         if y>0
                            y = y-1*fine;
                            redraw
%                         end
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
                    case 'escape'
                         esc = true;
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
                x_idx = round((x+1)/scale):size(im2,1)+round((x+1)/scale) - 1;
                y_idx =  round((y+1)/scale):size(im2,2)+round((y+1)/scale) - 1;
                mn_x =  abs(min([x_idx-1,0]));
                mn_y = abs(min([y_idx-1,0]));
                im3 = zeros(size(im1,1)+mn_x,size(im1,2)+mn_y,3);
                im3((1:size(im1,1))+mn_x,(1:size(im1,2))+mn_y,1) = im1;
                im3(x_idx+mn_x,y_idx+mn_y,2) = im2;
                set(gcf,'name',['X:' num2str(x/params.resize) ' Y:' num2str(y/params.resize) 'rot:' num2str(rot) ' scale:' num2str(scale)])
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
        
        function fim = filterImage(im,tform,inv)%{
            
            tform = projective2d(tform([5 4 7;2 1 8;3 6 9])); % correct for flipped image axis in matlab
            if nargin>2 && inv
                tform = invert(tform);
            end
            fim = imwarp(im,tform);
        end
        
        function im = processImage(im,varargin)
            params.prctile = 99;
            params.normalize = 1;
            params.exp = 1;
            
            params = ne7.mat.getParams(params,varargin);
            
            im(im>prctile(im(:),params.prctile)) = prctile(im(:),params.prctile);
            
            if params.exp
                im = ne7.mat.normalize(im).^params.exp;
            end
            
            if params.normalize
                im = ne7.mat.normalize(im);
            end
            
        end
        
        function h = image3D(x,y,z,I,IA,scale)
            I = flipud(I);
            IA = flipud(IA);
            
            if nargin<6; scale = 1; end
            
            %# coordinates
            [X,Y] = meshgrid(1:size(I,2), 1:size(I,1));
            Z = ones(size(I,1),size(I,2))*z;
            X = X - size(I,2)/2 + x;
            Y = Y - size(I,1)/2 - y;
            
            %# plot each slice as a texture-mapped surface (stacked along the Z-dimension)
            h = surface('XData',X*scale, 'YData',Y*scale, 'ZData',Z*scale, ...
                'CData',I, 'CDataMapping','direct', ...
                'EdgeColor','none', 'FaceColor','texturemap',...
                'AlphaData',IA,'FaceAlpha','texturemap','alphadatamapping','direct');
        end
    end
end