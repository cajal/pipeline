%{
vis.MovieSeqCond (manual) # random sequences of still frames
-> vis.Condition
---
-> vis.Movie
rng_seed                    : int                           # random number generator seed
pre_blank_period            : float                         # (s)
duration                    : float                         # (s) of each still
seq_length                  : smallint                      # number of frames in the sequence
movie_still_ids             : blob                          # sequence of stills
%}


classdef MovieSeqCond < dj.Relvar
end