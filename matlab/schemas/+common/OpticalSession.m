%{
common.OpticalSession (manual) # intrinsic imaging session
-> common.Animal
opt_sess  :   smallint      # optical session number for this animal
-----
opt_path  :   varchar(255)  # root path to raw data
opt_note  :   varchar(4095) # whatever you want
%}

classdef OpticalSession < dj.Relvar
    
    properties(Constant)
        table = dj.Table('common.OpticalSession')
    end
    
    methods
        function self = OpticalSession(varargin)
            self.restrict(varargin)
        end
    end
end