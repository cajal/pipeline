%{
pre.AlignRaster (computed) # raster alignment for bidirectional tuning
-> pre.ScanInfo
---
raster_phase                : float                         # shift of odd vs even raster lines
%}

classdef AlignRaster < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = pre.ScanInfo & (pre.ScanCheck & 'channel=1')
    end
    
    methods
        function fixRaster = get_fix_raster_fun(self)
            % returns a function that corrects the raster
            rasterPhase = self.fetch1('raster_phase');
            if rasterPhase == 0
                fixRaster = @(img) double(img);
            else
                fillFraction = fetch1(self*pre.ScanInfo, 'fill_fraction');
                fixRaster = @(img) ne7.ip.correctRaster(double(img), rasterPhase, fillFraction);
            end
        end
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            [template, bidirectional, fill_fraction] = fetch1(pre.ScanCheck*pre.ScanInfo & 'channel=1' & key, ...
                'template', 'bidirectional', 'fill_fraction');
            if bidirectional     % disabled raster correction temporarily
                taper = 10;  % the larger the number the thinner the taper
                sz = size(template);
                mask = atan(taper*hanning(sz(2)))'/atan(taper);
                im = bsxfun(@times, mask, template(:,:,1)-mean(mean(template(:,:,1))));
                key.raster_phase = ne7.ip.computeRasterCorrection(im, fill_fraction);
            else
                key.raster_phase = 0;
            end
            self.insert(key)
        end
    end
    
end
