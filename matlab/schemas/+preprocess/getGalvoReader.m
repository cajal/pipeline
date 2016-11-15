function reader = getGalvoReader(key)

[path, filename] = fetch1(experiment.Session*experiment.Scan & key, ...
    'scan_path', 'filename');

fprintf('Loading from %s\n', path);
path = getLocalPath(fullfile(path, sprintf('%s*.tif', filename)));

reader = ne7.scanimage.Reader5(path);

end