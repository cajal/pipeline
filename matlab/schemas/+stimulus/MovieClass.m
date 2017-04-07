%{
# types of movies 
movie_class : varchar(16)
%}

classdef MovieClass < dj.Lookup
    methods
        function fill(self)
            self.inserti({
                'mousecam'
                'object3d'
                'madmax'
                'multiobjects'})
        end
    end
end