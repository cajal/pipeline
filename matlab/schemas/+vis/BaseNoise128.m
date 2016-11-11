%{
vis.BaseNoise128 (lookup) #   base noise movies for generating stimuli of size 128 x 128
noise_seed  :  smallint   #  randon number generator seed
-----
noise_image : longblob    # 128 x 128
%}

classdef BaseNoise128 < dj.Relvar
    methods
        function fill(self)
            nx = 128;
            ny = 128;
            for seed = 1:1600
                r = RandStream.create('mt19937ar','NormalTransform', ...
                    'Ziggurat', 'Seed', seed);
                movie = int8(r.randn(ny, nx)/3*128);
                self.insert(struct('noise_seed', seed, 'noise_image', movie))
            end
        end
    end
end