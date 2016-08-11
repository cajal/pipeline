function data = getData( obj,channel )

if nargin<2
    channel = 1;
end

import vis2p.*

assert( length(obj)==1 );
[path,name] = fetch1( Experiments*obj, 'directory','file_name' );

scan_prog = fetch1(obj,'scan_prog');
if strcmp(scan_prog,'MPScan')
    filename = getLocalPath([path '/' name 'p%u.h5']);
    tpr = tpReader( filename );
    data = tpr.imCh{channel}(:,:,:);
elseif strcmp(scan_prog,'ScanImage')
    filename = getLocalPath([path '/' name]);
    tpr = tpMethods.Reader(filename);
    data = read(tpr,1);
else
    disp 'Can not read!'
    data = [];
end