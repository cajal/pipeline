%{
rf.AlignRaster (computed) # raster correction for resonant bidirectional scanning
-> rf.ScanCheck
-----
raster_phase =0             : float                         # shift of odd vs even raster lines
%}

classdef AlignRaster < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = rf.ScanCheck
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            [avg_frame, bidirectional, fill_fraction] = fetch1(rf.ScanCheck*rf.ScanInfo & key, ...
                'avg_frame', 'bidirectional', 'fill_fraction');
            if bidirectional
                key.raster_phase = ne7.ip.computeRasterCorrection(avg_frame, fill_fraction);
            end
            self.insert(key)
        end
    end
    
end