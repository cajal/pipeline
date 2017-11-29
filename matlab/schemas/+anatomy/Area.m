%{
# LUT for Brain Areas
brain_area      :  varchar('V1','P','POR','PM','AM','A','RL','AL','LI','LM')      # area name
---
%}

classdef Area < dj.Lookup
	methods
		function self = Area(varargin)
			self.restrict(varargin{:})
		end
    end
end