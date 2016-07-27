function reader = getReader(key)

[path, basename, scanIdx] = fetch1(rf.Session*rf.Scan & key, ...
    'scan_path', 'file_base', 'scan_idx');

fprintf('Loading from %s\n', path);
path = getLocalPath(fullfile(path, sprintf('%s_*%03u_*.tif', basename, scanIdx)));
reader = ne7.scanimage.Reader5(path);
end