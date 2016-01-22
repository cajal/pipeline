%{
pre.AlignRaster (computed) # raster alignment for bidirectional tuning
-> pre.ScanCheck
-----
raster_phase             : float                         # shift of odd vs even raster lines
%}

classdef AlignRaster < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = pre.ScanCheck
    end
    
    methods
        function fixRaster = get_fix_raster_fun(self)
            % returns a function that corrects the raster
            rasterPhase = self.fetch1('raster_phase');
            [fillFraction, nslices] = fetch1(self*pre.ScanInfo, ...
                'fill_fraction', 'nslices');
            assert(nslices==1, 'adjust this code to handle multiple slices')
            if rasterPhase == 0
                fixRaster = @(img) img;
            else
                fixRaster = @(img) ne7.ip.correctRaster(img, rasterPhase, fillFraction);
            end
        end
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            [template, bidirectional, fill_fraction] = fetch1(pre.ScanCheck*pre.ScanInfo & key, ...
                'template', 'bidirectional', 'fill_fraction');
            if bidirectional
                key.raster_phase = ne7.ip.computeRasterCorrection(template, fill_fraction);
            end
            self.insert(key)
        end
    end
    
end