%{
common.OpticalMovie (manual) # intrinsic imaging movie$
-> common.OpticalSession
opt_movie       : smallint               # optical movie id within the optical session
---
purpose=null                : enum('structure','stimulus','bar') # purpose of movie
filename                    : varchar(255)                  # filename of the movie
lens                        : float                         # objective lens magnification
fov                         : float                         # field of view: microns across the image (assume isometric pixels)
setup                       : varchar(255)                  # camera, recording software, etc
x=null                      : float                         # (um) position from manipulator
y=null                      : float                         # (um) position from manipulator
z=null                      : float                         # (um) position from manipulator
opt_movie_notes             : varchar(4095)                 # write a haiku poem here
%}

classdef OpticalMovie < dj.Relvar

	properties(Constant)
		table = dj.Table('common.OpticalMovie')
	end

	methods
		function self = OpticalMovie(varargin)
			self.restrict(varargin)
		end
	end
end
