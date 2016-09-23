%{
preprocess.CircleMap (computed) # my newest table
# add primary key here
-> preprocess.Eye
-> preprocess.CircleRadii
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
		popRel = preprocess.Eye*preprocess.CircleRadii ; % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
            rv = preprocess.CirclesLookup & struct('radius',key.radius) ;
            assert(count(rv)>0, 'Lookup table for this radius does not exist in database, create it using buildCirlceLookupTables') ;
            tp = fetch(rv) ;
            xdims = arrayfun(@(z) z.width, tp) ;
            ydims = arrayfun(@(z) z.height, tp) ;
            area = xdims.*ydims ;
            [~, idx] = max(area) ; % pick the lookup table with the largest area, arbitrary criteria
            width = tp(idx).width ;
            height = tp(idx).height ;
            rv = preprocess.CirclesLookup & struct('radius',key.radius,'width',width,'height',height) ;
            basepath = char(fetchn(rv, 'basepath')) ;
            lut = circle.assigncircles(basepath, width, height, key.radius) ;
            lut.getCircles() ;
            [x,y] = extractEyePos(key.animal_id,key.session, key.scan_idx) ;
            key.xrange = round([prctile(x,0.05) prctile(x,99.95)]) ;
            key.yrange = round([prctile(y,0.05) prctile(y,99.95)]) ;
            cbuf = sprintf('Pixel Range does not fit in a byte: animal_id=%d, session=%d, scanidx=%d, xrange=%d yrange=%d\n', key.animal_id, key.session, key.scan_idx, diff(key.xrange), diff(key.yrange));
            assert((diff(key.xrange)<width) && (diff(key.yrange)<height) && ~isempty(x),cbuf) ;
            key.map = fillcirclemembership(x,y,key.xrange,key.yrange,lut) ;
			self.insert(key)
		end
    end
end