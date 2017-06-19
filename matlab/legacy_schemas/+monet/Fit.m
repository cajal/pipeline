%{
monet.Fit (computed) # fitting RFs for computing significance
-> monet.CleanRF
-----
x  : float   # (degrees) RF location away from center
y  : float   # (degrees) RF location away from center
radius : float  # (degrees) geometric mean of major and minor axes
aspect : float  # aspect ratio
orientation : float #(degrees)  orientation of major axis
contrast  : float  # (in sigmas)
%}

classdef Fit < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = monet.CleanRF;
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            [width,height] = fetch1(monet.RF & key, 'degrees_x', 'degrees_y');
            map = fetch1(monet.CleanRF & key, 'clean_map');
            [key.x, key.y, key.radius, key.aspect, key.orientation, key.contrast] = ...
                fitBlob(mean(map(:,:,2:3),3), width, height);
            self.insert(key)
        end
    end
    
end




function [x,y,radius,aspect,theta,contrast] = fitBlob(img, width, height)
img = double(img);
ys = linspace(-height/2,+height/2,size(img,1));
xs = linspace( -width/2, +width/2,size(img,2));
[yi, xi] = ndgrid(ys,xs);

imagesc(xs,ys,img,[-1 1]*0.1);
colormap(ne7.vis.doppler)
axis image
img = abs(img);

[contrast,i] = max(img(:));
y = yi(i);
x = xi(i);

max_aspect = 4;
lb = [-width/2, -height/2, 2     ,-log(max_aspect), -pi, 0];
ub = [+width/2, +height/2, height,+log(max_aspect), +pi, contrast];
fn = @(a) loss(xi,yi,img,a);
a = [x y 5 0 0 0.8*contrast];
a = fmincon(fn, a, [],[],[],[], lb, ub, [], optimset('display','off'));


% assign outputs
x = a(1);
y = a(2);
radius = a(3);
aspect = exp(abs(a(4)));
theta = a(5);
if a(4)<0
    theta = theta + pi/4;
end
theta = mod(theta, pi);
contrast = a(6);

% plot for debugging
G = fit(xi,yi,x,y,radius*sqrt(aspect),radius/sqrt(aspect),theta,contrast);
hold on
[~,h] = contour(xs,ys,G,logspace(-3,-1,10),'k');
hold off
set(h,'LineWidth',0.25, 'LineColor',[1 1 1]*0.5)
xlabel 'azimith (degrees)'
ylabel 'elevation (degrees)'
title(sprintf('ScoreMixin = %0.4f\n', radius*contrast^1.5));
drawnow;

end


function L = loss(xi, yi, img, a)
assert(length(a)==6)
D = img - fit(xi,yi,a(1),a(2),a(3)*exp(+a(4)/2),a(3)*exp(-a(4)/2),a(5),a(6));
D = D-mean(D(:));
D = D.*(1+3*(D>0));   % positive difference is penalized more
L = sum(D(:).^2);
end


function G = fit(xi,yi,x,y,ax1,ax2,theta,contrast)
xr = cos(theta)*(xi-x) + sin(theta)*(yi-y);
yr = cos(theta)*(yi-y) - sin(theta)*(xi-x);
G = contrast*exp(-(xr/ax1).^2/2-(yr/ax2).^2/2);
end
