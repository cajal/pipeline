%{
pre.Channel (lookup) # two-photon channels
channel: tinyint # channel number 1=green, 2=red'
-----
%}

classdef Channel < dj.Relvar
    methods
        function fill(self)
            self.inserti({1})
            self.inserti({2})
        end
    end
end