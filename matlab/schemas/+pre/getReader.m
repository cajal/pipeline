function reader = getReader(key, cache_dir)

[path, basename, scanIdx] = fetch1(rf.Session*rf.Scan & key, ...
    'scan_path', 'file_base', 'scan_idx');

if nargin <= 1 || ~exist(cache_dir, 'file')==7
    fprintf('Loading from %s\n', path);
    path = getLocalPath(fullfile(path, sprintf('%s_%03u.tif', basename, scanIdx)));
    reader = ne7.scanimage.Reader4(path);
else
    [~, stump] = fileparts(path);
    cache_path = fullfile(cache_dir, stump);
    try
        fprintf('Loading from %s\n', cache_path);
        tpath = getLocalPath(fullfile(cache_path, sprintf('%s_%03u.tif', basename, scanIdx)));
        reader = ne7.scanimage.Reader4(tpath);
    catch
        from = fullfile(getLocalPath(path), sprintf('%s_%03u*.tif', basename, scanIdx));
        fprintf('Copying %s to %s\n', from, cache_path);
        copyfile(from, cache_path)
        path = getLocalPath(fullfile(cache_path, sprintf('%s_%03u.tif', basename, scanIdx)));
        reader = ne7.scanimage.Reader4(path);
    end
end