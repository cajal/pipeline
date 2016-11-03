%{
preprocess.CirclesLookup (computed) # my newest table
# add primary key here
-> preprocess.CircleRadii
-----
# add additional attributes
width       :   smallint        # width of the reactangle over which the map is
                                # created
height      :   smallint        # height of rectangle
basepath    :   varchar(4096)   # base path of where the map is stored
filename    :   varchar(256)    # filename in which map is stored
%}

classdef CirclesLookup < dj.Relvar & dj.AutoPopulate

	properties
		popRel=preprocess.CircleRadii ;
	end

	methods(Access=protected)

		function makeTuples(self, key)
            width = 512 ; % because the lookup matrix is a matrix of unsigned bytes, the maximum number here is 255
            height = 512 ;
            basepath = '/mnt/lab/home/atlab/pipeline/pipeline/matlab/scripts/Eye_Movements/CircleMaps' ;
            obj = circle.assigncircles(basepath, width, height, key.radius) ;
            obj.run() ;
            key.width = width ;
            key.height = height ;
            key.basepath = basepath ;
            key.filename = obj.filename ;
			self.insert(key)
		end
    end
end