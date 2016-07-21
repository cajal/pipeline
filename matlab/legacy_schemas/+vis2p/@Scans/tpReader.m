function tpr = tpReader( obj )

import vis2p.*

assert( length(obj)==1 );
[path,name] = fetch1( Experiments*obj, 'directory','file_name' );

scan_prog = fetch1(obj,'scan_prog');
if strcmp(scan_prog,'MPScan')
    filename = getLocalPath([path '/' name 'p%u.h5']);
    tpr = tpReader( filename );
elseif strcmp(scan_prog,'ScanImage')
    filename = getLocalPath([path '/' name]);
    tpr = tpMethods.Reader(filename);
elseif strcmp(scan_prog,'AOD')
    aim = fetch1(obj,'aim');
    if strcmp(aim,'stack'); type = 'Volume'; else type = 'Functional';end
    dirs = dir(['M:\Mouse\' name(1:10) '*']);
    names = vertcat(dirs.name);
    timediff = str2num(names(:,[12 13 15 16 18 19]))- str2num( name([12 13 15 16 18 19]));
    timeind = find(timediff<0);
    [~,i] = max(timediff(timeind));
    itime = timeind(i);
    filename = ['M:\Mouse\' dirs(itime).name '\AODAcq\' name];
    tpr = aodReader(filename,type);
else
    disp 'Can not read!'
    tpr = [];
end