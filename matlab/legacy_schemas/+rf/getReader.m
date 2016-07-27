function reader = getReader(key)
% Returns reader object for key. Key can be a key struture or 
% any relvar where fetch(rf.Align & key) returns a single tuple.

if ~isstruct(key)
    key = fetch(rf.Align & key);
end

assert(length(key) == 1, 'one scan at a time please')

% Fetch path and basename from TpSession and TpScan
[path, basename, scanIdx] = fetch1(...
    rf.Session*rf.Scan & key, ...
    'scan_path', 'file_base', 'scan_idx');

% Manually override path if using an external drive, etc
[~,hostname] = system('hostname'); 
hostname = hostname(1:end-1);
if strcmp(hostname,'JakesLaptop')
    path = ['V:\Two-Photon\Jake\' path(end-5:end) '\'];
end

reader = reso.reader(path,basename,scanIdx);
