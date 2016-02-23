function filename = getFilename( obj )
assert( length(obj)==1 );
[path,name] = fetch1( Experiments*obj, 'directory','file_name' );

if strcmp(fetch1(obj,'scan_prog'),'MPScan')
    filename = getLocalPath([path '/' name 'p%u.h5']);
elseif strcmp(fetch1(obj,'scan_prog'),'ScanImage')
    filename = getLocalPath([path '/' name '.tif']);
end

