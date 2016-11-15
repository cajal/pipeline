% setup circles lookup tables
% lookup tables are in files that are pointed to by a database table
% circles_lookup
% a unique file exists for a combination of width, height and radius
% if a previous file exists, it will be overwritten
% assumes datajoint is setup in the environment, and table definition is
% already created

radiuslist = [5:1:20] ; % need files for these radii
width = 512 ; % because the lookup matrix is a matrix of unsigned bytes, the maximum number here is 255
height = 512 ;
basepath = '/mnt/lab/home/atlab/pipeline/pipeline/matlab/scripts/Eye_Movements/CircleMaps' ;
mypool = parpool(16) ;
lt = {} ;
tuple = {} ;
obj = {} ;
map_x = {} ;
map_y = {} ;

% create objects ahead of time so that parfor can be used
for ii=1:length(radiuslist)
    obj{ii} = circle.assigncircles(basepath, width, height, radiuslist(ii)) ;
    tuple{ii}.radius = 0 ;
    tuple{ii}.width = 0 ;
    tuple{ii}.height = 0 ;
    tuple{ii}.basepath = '' ;
    tuple{ii}.filename = '';
end


try
    parfor ii=1:length(radiuslist)
        disp(sprintf('Processing radius=%d\n', radiuslist(ii))) ;
        obj{ii}.run() ; % build the file
        tp = fetch(preprocess.CirclesLookup & struct('width',width,'height',height,'radius',radiuslist(ii))) ;
        if (isempty(tp)) % make a database entry if it does not exist
            tuple{ii}.radius = radiuslist(ii) ;
            tuple{ii}.width = width ;
            tuple{ii}.height = height ;
            tuple{ii}.basepath = basepath ;
            tuple{ii}.filename = obj{ii}.filename;
            insert(preprocess.CirclesLookup, tuple{ii}) ;
        end
        delete(obj{ii}) ;
    end
    delete(mypool) ;
catch error
    disp(error) ;
    delete(mypool) ;
end
