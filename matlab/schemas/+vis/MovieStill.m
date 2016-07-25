%{
vis.MovieStill (lookup) # cached still frames from the movie
-> vis.Movie
still_id        : int                    # ids of still images from the movie
---
still_frame                 : longblob                      # uint8 grayscale movie
%}


classdef MovieStill < dj.Relvar
end