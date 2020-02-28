%{
# Optical Flow for all frames using Lucas-Kanade method
-> movies.OpticFlowOpt
-> stimulus.MovieClip
---
orientation          : mediumblob   # mean phase angle of optical flow
magnitude            : mediumblob   # Magnitude of optical flow
%}

classdef  OpticFlowFine < dj.Computed
    
    properties
        keySource  = stimulus.MovieClip * (movies.OpticFlowOpt & 'process = "yes"')
    end
    
    methods(Access=protected)
        function makeTuples(self,key) %create clips
            [rsz,algorithm,params] = fetch1(movies.OpticFlowOpt & key,'rsz','algorithm','params');
            
            % get video file
            filename = exportMovie(stimulus.MovieClip & key);
            vidReader = VideoReader(filename{1});
            
            % construct optic flow algorithm
            opticFlow = eval(sprintf('%s(%s)',algorithm,params));
            
            % initialize
            iframe = 0;
            nframes =  floor(vidReader.Duration*vidReader.FrameRate);
            sz = ceil([vidReader.Height vidReader.Width]/rsz);
            Orientation = nan(sz(1),sz(2),nframes);
            Magnitude = Orientation;
            
            % fun for each frame
            while hasFrame(vidReader)
                iframe = iframe+1;
                frame = imresize(readFrame(vidReader),sz);
                flow = estimateFlow(opticFlow,frame(:,:,1));
                Orientation(:,:,iframe) = flow.Orientation;
                Magnitude(:,:,iframe) = flow.Magnitude;
            end
            
            % insert key
            key.orientation = single(Orientation(:,:,1:iframe));
            key.magnitude =single( Magnitude(:,:,1:iframe));
            insert( self, key );
            
            % cleanup
            delete(filename{1})
        end
    end
    
    methods
        function play(self,export)
            
            if nargin<2
                export = false;
            end
            
            %fetch stuff
            filename = exportMovie(stimulus.MovieClip & self);
            vidReader = VideoReader(filename{1});
            [Orientation,Magnitude] = fetch1(self,'orientation','magnitude');
            sz = size(Orientation);
            rsz = fetch1(movies.OpticFlowOpt & self,'rsz');
            
            if export
                [~,name] = fileparts(filename);
                vw = VideoWriter(sprintf('%s_of',name),'MPEG-4');
                vw.FrameRate = vidReader.FrameRate;
                open(vw)
            end
            
            % setup figure
            figure
            set(gcf,'position',[300 200 1000 600])
            colormap gray
            [x,y] = meshgrid(1:sz(2),1:sz(1));
            
            % loop through frames
            for i = 1:sz(3)
                clf
                image(readFrame(vidReader))
                hold on
                quiver(x*rsz- (rsz-1),y*rsz - (rsz-1),cos(Orientation(:,:,i)).*Magnitude(:,:,i),...
                    sin(Orientation(:,:,i)).*Magnitude(:,:,i))
                set(gca,'YDir','reverse')
                axis off
                axis image
                xlim([-rsz/2 sz(2)*rsz+rsz/2])
                ylim([-rsz/2 sz(1)*rsz+rsz/2])
                drawnow
                if export
                   writeVideo(vw, getframe);
                end
            end
            
            % cleanup
            delete(filename{1})
            if export
                close(vw)
            end
        end
        
        function avgFlow(self)
            
            keys = fetch(self);
            X = [];Y = [];
            for ikey = 1:length(keys)
                key = keys(ikey);
                [Orientation,Magnitude] = fetch1(movies.OpticFlowFine & key,'orientation','magnitude');
                X{ikey} = mean(cos(Orientation).*Magnitude,3);
                Y{ikey} = mean(sin(Orientation).*Magnitude,3);
            end
            X = mean(reshape(cell2mat(X),size(X{1},1),size(X{1},2),[]),3);
            Y = mean(reshape(cell2mat(Y),size(Y{1},1),size(Y{1},2),[]),3);
            
            % setup figure
            figure
            sz = size(Orientation);
            [x,y] = meshgrid(1:sz(2),1:sz(1));
            quiver(x,y,X,Y)
            set(gca,'YDir','reverse')
            axis off
            axis image
            shg
        end
    end
end