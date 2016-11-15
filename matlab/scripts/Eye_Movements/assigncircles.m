% assign circles to each point in a rectangle
% for each point in the matirx, create a list of circle centers that it
% belongs to
% save the file at base path
% filename = imagewidth_imageheight_radius.mat

classdef assigncircles < handle
    
    
    properties
        width = 0 ;
        height = 0 ;
        radius = 0 ;
        filename = '' ;
        map_x = [] ;
        map_y = [] ;
    end
    
    methods
        
        function self = assigncircles(basepath, imagewidth, imageheight, radius)
           
            self.width = imagewidth ;
            self.height = imageheight ;
            self.radius = radius ;
            self.filename = sprintf('%s%s%d_%d_%d.bin', basepath, filesep, imagewidth, imageheight, radius) ;
        end
        
        
        function run(self)
            
            imagewidth = self.width ;
            imageheight = self.height ;
             
            ac_circles_x = {} ;
            ac_circles_y = {} ;
            
            if (imagewidth > 255) || (imageheight > 255)
                disp('Maximum width and height is 255') ;
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
                                cv_x(count) = uint8(kk) ;
                                cv_y(count) = uint8(ll) ;
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

            
            ac_circlesarray_x = uint8(zeros(imagewidth,imageheight,maxlen)) ;
            ac_circlesarray_y = uint8(zeros(imagewidth,imageheight,maxlen)) ;
           
            for ii=1:imagewidth
                for jj=1:imageheight
                    cv = ac_circles_x{jj,ii} ;
                    ac_circlesarray_x(jj,ii,1:length(cv))=cv ;
                    cv = ac_circles_y{jj,ii} ;
                    ac_circlesarray_y(jj,ii,1:length(cv))=cv ;
                end
            end
            
            fp = fopen(self.filename, 'w') ;
            if (fp > 0)
                fwrite(fp, ac_circlesarray_x, 'uint8') ;
                fwrite(fp, ac_circlesarray_y, 'uint8') ;
                fclose(fp) ;
            else
                disp('Unable to open file, check if lab share is mounted\n') ;
            end
        end
        
%         function getCircles(self)
%             disp(self.filename) ;
%             fp = fopen(self.filename, 'rb') ;
%             if (fp > 0)
%                 tv = fread(fp) ;
%                 maxlen = length(tv)/2/self.width/self.height ;
%                 if (maxlen - int32(maxlen) == 0)
%                     self.map_x = fp(1:length(tv)/2) ;
%                     self.map_y = fp((length(tv)/2)+1:end) ;
%                     self.map_x = reshape(self.map_x, self.height, self.width, maxlen) ;
%                     self.map_y = reshape(self.map_y, self.height, self.width, maxlen) ;
%                 else
%                     disp('File format is inconsistent with a 3D array\n') ;
%                 end
%             else
%                disp('File does not exist, check full filename and if lab share is mounted\n') ;
%             end
%         end
    end   
end
