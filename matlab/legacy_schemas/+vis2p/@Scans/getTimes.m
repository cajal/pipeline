function times = getTimes(obj) 


fnm = fetch1(obj,'file_name');
dr = fetch1(Experiments.*obj,'directory');
files = dir([dr '/' fnm  '_ts*']);
if isempty(files)
    timestamp = GetFileTime([dr '/' fnm 'p0.h5']); 
    save([dr '/' fnm  '_ts'],'timestamp');
    times = timestamp.Creation;
else
    tm = load([dr '/' files.name]);
    times = tm.timestamp.Creation;
end
