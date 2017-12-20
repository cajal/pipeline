%{
# Retinotopy for different directions
-> map.RetMap
-> map.OptImageBar
---
%}

classdef RetMapScan < dj.Imported
    methods(Access=protected)
        function makeTuples(obj,key) %create ref map
             insert( obj, key );
        end
    end
end

