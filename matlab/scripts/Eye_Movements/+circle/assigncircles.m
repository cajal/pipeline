% assign circles to each point in a rectangle
% for each point in the matirx, create a list of circle centers that it
% belongs to
% save the file at base path
% filename = imagewidth_imageheight_radius.bin

classdef assigncircles < handle
    
    
    properties
        width = 0 ;
        height = 0 ;
        radius = 0 ;
        basepath = '' ;
        filename = '' ;
        map_x = [] ;
        map_y = [] ;
    end
    
    methods
        
        function self = assigncircles(basepath, imagewidth, imageheight, radius)   
            self.width = imagewidth ;
            self.height = imageheight ;
            self.radius = radius ;
            self.basepath = basepath ;
            self.filename = sprintf('%d_%d_%d.bin', imagewidth, imageheight, radius) ;
        end
        
        
        function run(self)
            imagewidth = self.width ;
            imageheight = self.height ;
            filename = sprintf('%s%s%s', self.basepath, filesep, self.filename) ;
            
            ac_circles_x = {} ;
            ac_circles_y = {} ;
            
            if (imagewidth > 65535) || (imageheight > 65535)
                disp('Maximum width and height is 65535') ;
                return ;
            end ;
            
            for ii=1:imagewidth
                for jj=1:imageheight
                    count = 0 ;
                    cv_x = [] ;
                    cv_y = [] ;
                    for kk=1:imagewidth
                        for ll=1:imageheight
                            if (ii-kk)^2+(jj-ll)^2<self.radius^2
                                count = count+1 ;
                                cv_x(count) = uint16(kk) ;
                                cv_y(count) = uint16(ll) ;
                            end 
                        end 
                    end 
                    ac_circles_x{jj,ii} = cv_x ;
                    ac_circles_y{jj,ii} = cv_y ;
                end 
            end 
            
            maxlen = 0 ;
            for ii=1:imagewidth
                for kk=1:imageheight
                    cv = ac_circles_x{kk,ii} ;
                    if (length(cv) > maxlen)
                        maxlen = length(cv) ;
                    end
                end
            end

            
            ac_circlesarray_x = uint16(zeros(imageheight,imagewidth,maxlen)) ;
            ac_circlesarray_y = uint16(zeros(imageheight,imagewidth,maxlen)) ;
           
            for ii=1:imagewidth
                for jj=1:imageheight
                    cv = ac_circles_x{jj,ii} ;
                    ac_circlesarray_x(jj,ii,1:length(cv))=cv ;
                    cv = ac_circles_y{jj,ii} ;
                    ac_circlesarray_y(jj,ii,1:length(cv))=cv ;
                end
            end
            
            fp = fopen(filename, 'w') ;
            if (fp > 0)
                fwrite(fp, ac_circlesarray_x, 'uint16') ;
                fwrite(fp, ac_circlesarray_y, 'uint16') ;
                fclose(fp) ;
            else
                disp('Unable to open file, check if lab share is mounted\n') ;
            end
        end
        
        function getCircles(self)
            filename = sprintf('%s%s%s', self.basepath, filesep, self.filename) ;
            fp = fopen(filename, 'rb') ;
            if (fp > 0)
                tv = uint16(fread(fp, 'uint16')) ;
                maxlen = length(tv)/2/self.width/self.height ;
                if (maxlen - int32(maxlen) == 0)
                    self.map_x = uint16(tv(1:length(tv)/2)) ;
                    self.map_y = uint16(tv((length(tv)/2)+1:end)) ;
                    self.map_x = permute(reshape(self.map_x, self.width, self.height, maxlen),[2 1 3]) ;
                    self.map_y = permute(reshape(self.map_y, self.width, self.height, maxlen),[2 1 3]) ;
                else
                    disp('File format is inconsistent with a 3D array\n') ;
                end
            else
                disp('File does not exist, check full filename and if lab share is mounted\n') ;
            end
        end
        
        
        function circles = circlesforthispoint(self,x,y)
            idx = find(self.map_x(y,x,:)) ;
            xv = self.map_x(y,x,idx) ;
            yv = self.map_y(y,x,idx) ;
            for ii=1:length(xv)
                circles{ii}.x = yv(ii) ;
                circles{ii}.y = xv(ii) ;
            end ;
        end
        
    end   
end
