%{
# stacks with beads for PSF computations
# stacks with beads for PSF computations$
date                        : date                          # acquisition date
stack_num                   : smallint                      # stack number for that day
---
who                         : varchar(63)                   # who acquired the data
-> experiment.Rig
lens                        : decimal(5,2)                  # lens magnification
na                          : decimal(3,2)                  # numerical aperture of the objective lens
fov_x                       : float                         # (um) field of view at selected magnification
fov_y                       : float                         # (um) field of view at selected magnification
wavelength                  : smallint                      # (nm) laser wavelength
mwatts                  : decimal(3,1)                  # mwatts out of objective
path                        : varchar(1023)                 # file path
full_filename               : varchar(255)                  # file name
note                        : varchar(1023)                 # any other information
beadstack_ts=CURRENT_TIMESTAMP: timestamp                   # automatic
%}

classdef BeadStack < dj.Manual
end