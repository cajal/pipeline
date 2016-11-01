% extract eye positions for all animals, all sessions and all scans
% build a map, an element of the map represents the number of eye positions
% within a circle centered at the element
function BuildEyePositionClusterMap(radius)

% get lookup table to assign circle membership to eye positions
rv = preprocess.CirclesLookup & struct('radius',radius) ;
assert(count(rv)>0, 'Lookup table for this radius does not exist in database, create it using buildCirlceLookupTables') ;
tp = fetch(rv) ;
xdims = arrayfun(@(z) z.width, tp) ;
ydims = arrayfun(@(z) z.height, tp) ;
area = xdims.*ydims ;
[~, idx] = max(area) ; % pick the lookup table with the largest area, arbitrary criteria
width = tp(idx).width ;
height = tp(idx).height ;
rv = preprocess.CirclesLookup & struct('radius',radius,'width',width,'height',height) ;
basepath = char(fetchn(rv, 'basepath')) ;
lut = circle.assigncircles(basepath, width, height, radius) ;
lut.getCircles() ;

animals = fetchn(common.Animal & preprocess.EyeTrackingFrame, 'animal_id') ;

for ii=1:length(animals)
    rvsessions = preprocess.EyeTrackingFrame & struct('animal_id', animals(ii)) ;
    sessions = unique(fetchn(rvsessions, 'session')) ;
    for jj=1:length(sessions)
        rvscans = rvsessions & struct('session', sessions(jj)) ;
        scans = unique(fetchn(rvscans, 'scan_idx')) ;
        for kk=1:length(scans)
            tp = fetch(preprocess.CircleMap & struct('animal_id', animals(ii),...
                                                'session', sessions(jj), ...
                                                'scan_idx', scans(kk),...
                                                'radius', radius)) ;
            if (isempty(tp))
                [x,y] = extractEyePos(animals(ii),sessions(jj), scans(kk)) ;
                figure(kk) ;
                plot(x,y, '.') ;
                xrange = round([prctile(x,0.05) prctile(x,99.95)]) ;
                yrange = round([prctile(y,0.05) prctile(y,99.95)]) ;
                if (diff(xrange)<width) && (diff(yrange)<height) && ~isempty(x)
                    circlemembership = fillcirclemembership(x,y,xrange,yrange,lut) ;
%                 [p,q] = max(circlemembership) ;
%                 [~,s] = max(p) ;
%                peak = [q(s),s] ; % row,col
%                rf_includedeyepos = isValidEyePos(x,y,peak,radius) ;
%               write the map and the cluster peak to database
                    try
                        insert(preprocess.CircleMap, struct('animal_id', animals(ii),...
                                                'session', sessions(jj), ...
                                                'scan_idx', scans(kk),...
                                                'map', circlemembership,...
                                                'radius', radius,...
                                                'xrange', xrange,...
                                                'yrange', yrange)) ;
                    catch
                    end
                    figure(kk+100) ;
                    mesh(circlemembership) ;
                else
                    disp(sprintf('Pixel Range does not fit in a byte: animal_id=%d, session=%d, scanidx=%d, xrange=%d yrange=%d\n', animals(ii), sessions(jj), scans(kk), diff(xrange), diff(yrange)));
                end
            end
        end
    end
end
