%{
common.TpStack (manual) # my newest table
-> common.TpSession
stack_idx : smallint  #   stack number
-----
surfz  : float   # (um) z-coord at pial surface
laser_wavelength  : float # (nm)
laser_power :  float # (mW) to brain
stack_notes = "" : varchar(4095)  #  free-notes
stack_ts = CURRENT_TIMESTAMP  : timestamp  #  automatic
%}

classdef TpStack < dj.Relvar
    
    properties(Constant)
        table = dj.Table('common.TpStack')
    end
    
    methods
        function self = TpStack(varargin)
            self.restrict(varargin)
        end
        
        
        function [r,g,scim] = getStack(self)
            key = fetch(self);
            assert(length(key)==1, 'one stack at a time please')
            f = getFilename(common.TpScan & key,0,'stack');
            f=f{1};
            scim = ne7.scanimage.Reader(f);
            disp 'reading green channel...'
            g = scim.read(1);
            disp 'reading red channel...'
            r = scim.read(2);
        end
        
        
        function descend(self)
            key = fetch(self);
            assert(length(key)==1, 'one stack at a time please')
            key = fetch(self);
            assert(length(key)==1, 'one stack at a time please')
            f = getFilename(common.TpScan & key,0,'stack');
            f=f{1};
            scim = ne7.scanimage.Reader(f);
            
            slab = 200;
            height = 1200;
            
            oldix = [];
            fnum = 1;
            for iFrame = 1:10:scim.nFrames-slab;
                fprintf('frame %03d:', iFrame)
                fname = sprintf('frame%03d.png', fnum);
                ix = scim.nFrames - slab - iFrame+1 + (1:slab);
                if isempty(oldix)
                    tic
                    r = scim.read(2,ix);
                    g = scim.read(1,ix);
                    toc
                else
                    tic
                    n = length(setdiff(oldix,ix));
                    r(:,:,end-n+1:end) = [];
                    g(:,:,end-n+1:end) = [];
                    r = cat(3,scim.read(2,setdiff(ix,oldix)),r);
                    g = cat(3,scim.read(1,setdiff(ix,oldix)),g);
                    toc
                end
                oldix = ix;
                if ~exist(fname,'file')
                    rframe = projectPerspective(r, height);
                    fprintf |
                    gframe = projectPerspective(g, height);
                    img = max(0, min(1, cat(3,rframe,gframe,rframe*0)));
                    imwrite(imresize(img,0.5), fname, 'png')
                end
                fprintf \n
                fnum = fnum+1;
            end
        end
    end
end



function frame = projectPerspective(stack, height)
frame = 0;
sz = size(stack);
T = maketform('affine',[1 0 0; 0 1 0; 0 0 1]);
for iSlice=1:sz(3)
    if ~mod(iSlice,10)
        fprintf .
    end
    scale = (height+iSlice-1)/height;
    im = stack(:,:,iSlice);
    im = imtransform(im, T, 'bicubic', 'size', size(im), ...
        'udata',[-1 1], 'vdata',[-1 1], ...
        'xdata', [-1 1]/scale, 'ydata', [-1 1]/scale);
    frame = max(frame, im);
end
frame = frame - quantile(frame(:),0.01);
frame = frame / quantile(frame(:),0.998);
end
