%{
rf.Lens (lookup) # objective lens list
setup           : char(4)                # two-photon setup
lens            : char(4)                # objective lens
---
%}

classdef Lens < dj.Relvar
    methods
        function fill(self)
            self.inserti({
                '2P1'  '16x'
                '2P1'  '25x'
                '2P2'  '16x'
                '2P2'  '25x'
                '2P3'  '16x'
                '2P3'  '25x'
                })
        end
    end
end
