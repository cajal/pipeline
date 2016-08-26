%{
preprocess.MaskCoordinates (imported) # mask center of mass of a segmented cell
-> preprocess.ScanCoordinates
trace_id        : smallint               # trace id
-----
xloc                 : double # x location in micro meters relative to the frame start
yloc                 : double # y location in micro meters relative to the frame start
zloc                 : double # z location in micro meters relative to the surface
%}

classdef MaskCoordinates < dj.Relvar 
    
    properties
        popRel  = preprocess.ScanCoordinates
    end
    
    methods
        
        function makeTuples(self, key)
            
         
                self.insert(key);
           
        end
    end
end