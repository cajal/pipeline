%{
rf.NoiseMapMethod (lookup) # RF computation method for Gaussian noise stimuli
noise_map_method : tinyint  # noise map method number
-----
noise_map_algorithm  : enum('STA','linear','STPCA')  # algorithm for RF mapping
%}

classdef NoiseMapMethod < dj.Relvar
    methods
        function fill(self)
            self.inserti({
                1  'STA'
                2  'linear'
                3  'STPCA'
                })
        end
    end
end