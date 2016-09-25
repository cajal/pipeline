%{
preprocess.CircleMap (computed) # my newest table
# add primary key here
-> preprocess.Eye
-> preprocess.CirclesLookup
-----
# add additional attributes
map     :   mediumblob  # map of cluster membership, i.e. how many eye
                        # positions lie within a radius centered at each element in the map
xrange  :   tinyblob    # range of eye positions used in the map
                        # calculations, x direction, [lo, hi]
yrange  :   tinyblob    # range in y direction
%}

classdef CircleMap < dj.Relvar & dj.AutoPopulate

	properties
		popRel = preprocess.Eye*preprocess.CirclesLookup ; % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
            [x,y] = extractEyePos(key.animal_id,key.session, key.scan_idx) ;
            key.xrange = round([prctile(x,0.05) prctile(x,99.95)]) ;
            key.yrange = round([prctile(y,0.05) prctile(y,99.95)]) ;
            if ~isempty(x) && ~any(isnan(key.xrange)) && ~any(isnan(key.yrange))
                rv = preprocess.CirclesLookup & struct('radius',key.radius) ;
                [basepath,width,height] = rv.fetchn('basepath', 'width', 'height') ;
                lut = circle.assigncircles(char(basepath), width, height, key.radius) ;
                lut.getCircles() ;
                cbuf = sprintf('Pixel Range does not fit in the map range: animal_id=%d, session=%d, scanidx=%d, xrange=%d yrange=%d\n', key.animal_id, key.session, key.scan_idx, diff(key.xrange), diff(key.yrange));
                assert((diff(key.xrange)<width) && (diff(key.yrange)<height) && ~isempty(x),cbuf) ;
                key.map = fillcirclemembership(x,y,key.xrange,key.yrange,lut) ;
                self.insert(key)
            end
		end
    end
end