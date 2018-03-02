%{
# align a 2-d scan field to a stack
-> stack.RegistrationTask
---
reg_x                       : float                         # (px) center of scan in stack coordinates
reg_y                       : float                         # (px) center of scan in stack coordinates
reg_z                       : float                         # (um) depth of scan in stack coordinates
yaw=0                       : float                         # degrees of rotation over the z axis
pitch=0                     : float                         # degrees of rotation over the y axis
roll=0                      : float                         # degrees of rotation over the x axis
score                       : float                         # cross-correlation score (-1 to 1)
common_res                  : float                         # (um/px) common resolution used for registration
%}


classdef FieldRegistration < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end