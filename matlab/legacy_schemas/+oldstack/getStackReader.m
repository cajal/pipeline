function reader = getStackReader(key)

[path, filename] = fetch1(experiment.Session*experiment.Stack & key, 'scan_path', 'filename');

fprintf('Loading from %s\n', path);

path = getLocalPath(fullfile(path, sprintf('%s_*.tif', filename)));

reader = ne7.scanimage.Reader5(path);

end