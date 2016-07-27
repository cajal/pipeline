function startTime = fetchTime(obj)

assert( length(obj)==1 );
[path,name] = fetch1( Experiments*obj, 'directory','file_name' );
files = dir(getLocalPath([path '/' name  '_ts*']));
if isempty(files)
    timestamp = GetFileTime(getLocalPath([path '/' name 'p0.h5'])); 
    save([path '/' name  '_ts'],'timestamp');
    startTime = timestamp.Creation;
else
    tm = load(getLocalPath([path '/' files.name]));
    startTime = tm.timestamp.Creation;
end
